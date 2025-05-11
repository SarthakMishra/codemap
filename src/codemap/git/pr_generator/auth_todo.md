To implement a user-friendly authentication flow in your Python CLI tool-similar to how GitHub’s own CLI (`gh`) provides a login link and handles token storage-you should use GitHub’s OAuth Device Authorization flow. This method allows users to authenticate via a browser, grant access, and have your CLI automatically retrieve and store the access token.

## How the GitHub CLI (`gh`) Authentication Flow Works

- The CLI initiates the OAuth Device Authorization flow.
- The user is shown a URL and a one-time code in the terminal.
- The user visits the URL, enters the code, and logs in to GitHub to approve the app.
- The CLI polls GitHub’s API and, once authorized, receives an access token.
- The CLI stores the token locally for future API requests.

## How to Implement This in Your Python CLI

**1. Register an OAuth App with GitHub**
- Go to GitHub → Settings → Developer settings → OAuth Apps.
- Register a new OAuth application.
- Set the callback URL to something like `http://localhost` (not used in device flow, but required for registration).
- Note the Client ID and Client Secret.

**2. Implement the Device Authorization Flow**

Here’s a high-level outline:

- Request a device and user code from GitHub.
- Display the verification URL and user code to the user.
- Poll GitHub for the access token.
- Store the access token securely (e.g., in a config file).

**Example (simplified):**

```python
import requests
import time

CLIENT_ID = "your_client_id"

# Step 1: Request device and user codes
resp = requests.post(
    "https://github.com/login/device/code",
    data={"client_id": CLIENT_ID, "scope": "repo"}
)
resp.raise_for_status()
data = resp.json()
print(f"Visit {data['verification_uri']} and enter code: {data['user_code']}")

# Step 2: Poll for token
while True:
    time.sleep(data['interval'])
    token_resp = requests.post(
        "https://github.com/login/oauth/access_token",
        data={
            "client_id": CLIENT_ID,
            "device_code": data["device_code"],
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
        },
        headers={"Accept": "application/json"}
    )
    token_data = token_resp.json()
    if "access_token" in token_data:
        print("Authentication successful!")
        access_token = token_data["access_token"]
        # Save access_token securely for future use
        break
    elif token_data.get("error") != "authorization_pending":
        print("Error:", token_data.get("error_description"))
        break
```

**3. Use the Token in API Requests**

When making requests to the GitHub REST API, include the token in the `Authorization` header:

```python
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/vnd.github+json"
}
response = requests.get("https://api.github.com/user", headers=headers)
```


## Key Points

- The Device Authorization flow is designed for CLI and desktop apps, providing a secure and user-friendly authentication experience.
- This approach does not require users to manually create or paste tokens.
- Store the token securely (e.g., use OS keyring or encrypted file).

## References

- [GitHub REST API Authentication Docs][1]
- [GitHub CLI Manual][7]
- [Quickstart for GitHub REST API][2]

This flow will give your CLI tool an authentication experience very similar to the official `gh` CLI.