# Deploy GeoConflictModeler on Render

This repo includes a `render.yaml` Blueprint that provisions:
- Static site (`site/`)
- API (`api/`) for accounts + Stripe subscriptions + access tokens
- Streamlit app (`app/`)
- Postgres database

## 1) Create a Git repo
1. Create a GitHub repo.
2. Upload this folder contents.

## 2) Deploy via Blueprint
1. In Render Dashboard: **New** → **Blueprint**.
2. Select your repo.
3. Render will detect `render.yaml` and create all resources.

## 3) Configure environment variables

### API service (`geoconflictmodeler-api`)
Set these (in Render → Service → Environment):

- `JWT_SECRET` = long random string
- `APP_TOKEN_SECRET` = long random string
- `APP_BASE_URL` = your Streamlit app URL (example: `https://app.yourdomain.com`)

Stripe (when you’re ready):
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `STRIPE_PRICE_ID`

Checkout redirects (recommended):
- `CHECKOUT_SUCCESS_URL` = `https://YOUR_SITE_DOMAIN/success.html`
- `CHECKOUT_CANCEL_URL` = `https://YOUR_SITE_DOMAIN/cancel.html`

### App service (`geoconflictmodeler-app`)
By default, paywall is OFF (`GCM_REQUIRE_PAID=0`).

When you’re ready to require subscriptions:
- Set `GCM_REQUIRE_PAID=1`
- Set `GCM_API_BASE_URL` = your API base URL (example: `https://api.yourdomain.com`)

## 4) Custom domains
Recommended split:
- `geoconflictmodeler.com` → Static site
- `api.geoconflictmodeler.com` → API
- `app.geoconflictmodeler.com` → Streamlit app

Once you attach domains, update:
- `APP_BASE_URL` on the API service to your app domain
- (Optional) Set `SITE_ORIGIN` to your site domain instead of `*`

## 5) Stripe webhook
Create a webhook endpoint in Stripe:
- URL: `https://api.YOURDOMAIN.com/billing/webhook`
- Events:
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`

## 6) Test flow
1. Open the site → register/login
2. Subscribe → Stripe checkout
3. Return to `success.html`
4. Click Launch → redirected to Streamlit with short-lived token

Contact: **bessuman.academia@gmail.com**
