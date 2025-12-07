import os
from dotenv import load_dotenv
import dropbox

load_dotenv()

APP_KEY = os.getenv("DROPBOX_APP_KEY")
APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

flow = dropbox.DropboxOAuth2FlowNoRedirect(
    APP_KEY,
    APP_SECRET,
    token_access_type="offline"  # ← КРИТИЧЕСКИ важно
)

auth_url = flow.start()
print("1) Open this URL in browser:")
print(auth_url)
print("\n2) Click 'Allow'")
print("3) Copy the authorization code\n")

code = input("Paste authorization code here: ").strip()

oauth_result = flow.finish(code)

print("\n✅ SAVE THIS INTO .env AS DROPBOX_REFRESH_TOKEN:\n")
print(oauth_result.refresh_token)
