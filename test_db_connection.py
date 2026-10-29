import asyncio

from sqlalchemy import text

from app.db.database import engine, init_db


async def main():
    try:
        async with engine.connect() as connection:
            result = await connection.execute(text("SELECT 1"))
            print("Database connection:", result.scalar_one())

        await init_db()
        print("Database tables initialized.")

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())