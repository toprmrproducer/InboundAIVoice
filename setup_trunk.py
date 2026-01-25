import asyncio
import os
from dotenv import load_dotenv
from livekit import api

# Load environment variables
load_dotenv(".env")

async def main():
    # Initialize LiveKit API
    # Credentials (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) are auto-loaded from .env
    lkapi = api.LiveKitAPI()
    sip = lkapi.sip
    
    trunk_id = os.getenv("OUTBOUND_TRUNK_ID")
    address = os.getenv("VOBIZ_SIP_DOMAIN")
    username = os.getenv("VOBIZ_USERNAME")
    password = os.getenv("VOBIZ_PASSWORD")
    number = os.getenv("VOBIZ_OUTBOUND_NUMBER")
    
    if not trunk_id:
        print("Error: OUTBOUND_TRUNK_ID not found in .env")
        return

    print(f"Updating SIP Trunk: {trunk_id}")
    print(f"  Address: {address}")
    print(f"  Username: {username}")
    print(f"  Numbers: [{number}]")

    try:
        # Update the trunk with the correct credentials and settings
        await sip.update_outbound_trunk_fields(
            trunk_id,
            address=address,
            auth_username=username,
            auth_password=password,
            numbers=[number] if number else [],
        )
        print("\n✅ SIP Trunk updated successfully!")
        print("The 'max auth retry attempts' error should be resolved now.")
        
    except Exception as e:
        print(f"\n❌ Failed to update trunk: {e}")
    finally:
        await lkapi.aclose()

if __name__ == "__main__":
    asyncio.run(main())
