import json
import pandas as pd
import asyncio
from tqdm import tqdm
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest
from utils.paths import Paths


async def parse_channel(api_credentials: dict):
    client = TelegramClient(
        api_credentials["session_name"],
        api_credentials["api_id"],
        api_credentials["api_hash"]
    )

    await client.connect()

    if not await client.is_user_authorized():
        phone = api_credentials["phone"]
        code = api_credentials.get("code")
        password = api_credentials.get("password")

        if not code:
            raise ValueError("В конфиге не указан одноразовый код (`code`) из Telegram")
        
        await client.send_code_request(phone)
        try:
            await client.sign_in(phone=phone, code=code)
        except SessionPasswordNeededError:
            if not password:
                raise ValueError("Включена двухфакторная авторизация, но `password` не указан в конфиге")
            await client.sign_in(password=password)

    try:
        channel = await client.get_entity(api_credentials["channel_name"])
    except Exception:
        return []

    posts = []
    offset_id = 0
    limit = 100

    try:
        with tqdm(desc="Progress") as pbar:
            while True:
                history = await client(GetHistoryRequest(
                    peer=channel,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=limit,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))

                if not history.messages:
                    break

                for msg in history.messages:
                    if msg.message:
                        posts.append({
                            'id': msg.id,
                            'date': msg.date.isoformat(),
                            'text': msg.message,
                            'views': getattr(msg, 'views', None),
                            'forwards': getattr(msg, 'forwards', None)
                        })

                offset_id = history.messages[-1].id
                pbar.update(len(history.messages))
                await asyncio.sleep(1)

    finally:
        await client.disconnect()

    return posts


def run_parser(api_credentials: dict, paths: Paths):
    messages = asyncio.get_event_loop().run_until_complete(parse_channel(api_credentials))
    if not messages:
        return

    with open(paths.resolve("data/tg_messages.json"), 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    pd.DataFrame(messages).to_csv(paths.tg_messages, index=False)


if __name__ == "__main__":
    from utils.io import load_config
    config = load_config("configs/base.json")
    run_parser(config["telegram"], Paths(config))
