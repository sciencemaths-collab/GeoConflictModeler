from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import jwt
import stripe
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, String, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# ----------------------------
# Config
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-change-me")
APP_TOKEN_SECRET = os.getenv("APP_TOKEN_SECRET", "dev-change-me")
JWT_ALG = "HS256"

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")

SITE_ORIGIN = os.getenv("SITE_ORIGIN", "*")
APP_BASE_URL = os.getenv("APP_BASE_URL", "")  # e.g. https://app.geoconflictmodeler.com

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Stripe init (safe if keys empty; billing endpoints will error gracefully)
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# ----------------------------
# DB
# ----------------------------
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))

    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    subscription_status: Mapped[str] = mapped_column(String(32), default="none")  # none|active|past_due|canceled
    current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def _mk_engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return create_engine(DATABASE_URL, pool_pre_ping=True)

engine = None
SessionLocal = None

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="GeoConflictModeler API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[SITE_ORIGIN] if SITE_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

security = HTTPBearer(auto_error=False)


def _db():
    global engine, SessionLocal
    if SessionLocal is None:
        engine = _mk_engine()
        SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _jwt_encode(payload: dict, secret: str, exp_seconds: int) -> str:
    now = int(time.time())
    payload = dict(payload)
    payload.update({"iat": now, "exp": now + exp_seconds})
    return jwt.encode(payload, secret, algorithm=JWT_ALG)


def _jwt_decode(token: str, secret: str) -> dict:
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security), db=Depends(_db)) -> User:
    if creds is None:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    payload = _jwt_decode(creds.credentials, JWT_SECRET)
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Bad token")
    user = db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ----------------------------
# Schemas
# ----------------------------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    token: str

class CheckoutOut(BaseModel):
    url: str

class StatusOut(BaseModel):
    active: bool
    subscription_status: str
    current_period_end: Optional[str] = None

class MintOut(BaseModel):
    token: str
    app_url: str


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/auth/register", response_model=TokenOut)
def register(data: RegisterIn, db=Depends(_db)):
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing = db.query(User).filter(User.email == data.email.lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    u = User(
        id=str(uuid4()),
        email=data.email.lower(),
        password_hash=pwd_ctx.hash(data.password),
        subscription_status="none",
    )
    db.add(u)
    db.commit()

    tok = _jwt_encode({"sub": u.id, "email": u.email}, JWT_SECRET, exp_seconds=60 * 60 * 24 * 7)
    return TokenOut(token=tok)


@app.post("/auth/login", response_model=TokenOut)
def login(data: LoginIn, db=Depends(_db)):
    u = db.query(User).filter(User.email == data.email.lower()).first()
    if not u or not pwd_ctx.verify(data.password, u.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    tok = _jwt_encode({"sub": u.id, "email": u.email}, JWT_SECRET, exp_seconds=60 * 60 * 24 * 7)
    return TokenOut(token=tok)


@app.get("/me")
def me(user: User = Depends(require_user)):
    return {
        "id": user.id,
        "email": user.email,
        "subscription_status": user.subscription_status,
        "current_period_end": user.current_period_end.isoformat() if user.current_period_end else None,
    }


def _is_active(u: User) -> bool:
    return u.subscription_status == "active"


@app.get("/access/status", response_model=StatusOut)
def access_status(user: User = Depends(require_user)):
    return StatusOut(
        active=_is_active(user),
        subscription_status=user.subscription_status,
        current_period_end=user.current_period_end.isoformat() if user.current_period_end else None,
    )


@app.post("/access/mint", response_model=MintOut)
def mint(user: User = Depends(require_user)):
    if not _is_active(user):
        raise HTTPException(status_code=402, detail="Subscription required")

    if not APP_BASE_URL:
        raise HTTPException(status_code=500, detail="APP_BASE_URL not configured")

    app_tok = _jwt_encode({"sub": user.id, "scope": "play"}, APP_TOKEN_SECRET, exp_seconds=60 * 10)
    return MintOut(token=app_tok, app_url=f"{APP_BASE_URL}?token={app_tok}")


@app.get("/access/verify")
def verify(token: str, db=Depends(_db)):
    payload = _jwt_decode(token, APP_TOKEN_SECRET)
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Bad token")
    u = db.get(User, uid)
    if not u or not _is_active(u):
        raise HTTPException(status_code=401, detail="Not active")
    return {"ok": True, "sub": uid}


# ----------------------------
# Stripe Billing
# ----------------------------
@app.post("/billing/create-checkout", response_model=CheckoutOut)
def create_checkout(user: User = Depends(require_user), db=Depends(_db)):
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    # Create/reuse Stripe customer
    if not user.stripe_customer_id:
        customer = stripe.Customer.create(email=user.email, metadata={"user_id": user.id})
        user.stripe_customer_id = customer["id"]
        db.add(user)
        db.commit()

    success_url = os.getenv("CHECKOUT_SUCCESS_URL", "")
    cancel_url = os.getenv("CHECKOUT_CANCEL_URL", "")
    if not success_url or not cancel_url:
        raise HTTPException(status_code=500, detail="Missing CHECKOUT_SUCCESS_URL/CHECKOUT_CANCEL_URL")

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=user.stripe_customer_id,
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        client_reference_id=user.id,
        metadata={"user_id": user.id},
    )
    return CheckoutOut(url=session.url)


@app.post("/billing/webhook")
async def stripe_webhook(request: Request, db=Depends(_db)):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")

    payload = await request.body()
    sig = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig, secret=STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {e}")

    et = event.get("type")
    data = event.get("data", {}).get("object", {})

    def _set_user_status_by_customer(customer_id: str, status: str, sub_id: Optional[str], period_end: Optional[int]):
        u = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if not u:
            return
        u.subscription_status = status
        u.stripe_subscription_id = sub_id
        u.current_period_end = datetime.fromtimestamp(period_end, tz=timezone.utc) if period_end else None
        db.add(u)
        db.commit()

    if et == "checkout.session.completed":
        customer_id = data.get("customer")
        sub_id = data.get("subscription")
        if customer_id and sub_id:
            sub = stripe.Subscription.retrieve(sub_id)
            status = sub.get("status", "")
            period_end = sub.get("current_period_end")
            mapped = "active" if status == "active" else "past_due" if status in ("past_due", "unpaid") else "canceled"
            _set_user_status_by_customer(customer_id, mapped, sub_id, period_end)

    elif et in ("customer.subscription.created", "customer.subscription.updated"):
        customer_id = data.get("customer")
        sub_id = data.get("id")
        status = data.get("status", "")
        period_end = data.get("current_period_end")
        mapped = "active" if status == "active" else "past_due" if status in ("past_due", "unpaid") else "canceled"
        if customer_id:
            _set_user_status_by_customer(customer_id, mapped, sub_id, period_end)

    elif et in ("customer.subscription.deleted",):
        customer_id = data.get("customer")
        sub_id = data.get("id")
        if customer_id:
            _set_user_status_by_customer(customer_id, "canceled", sub_id, None)

    # You can extend this to react to invoice.payment_failed, etc.

    return {"received": True}
