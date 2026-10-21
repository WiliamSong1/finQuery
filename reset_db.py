
from app.database.vector_store import (
    get_client,
    COLLECTION_NAME
)


def reset_database():
    client = get_client()

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(
            collection_name=COLLECTION_NAME
        )
        print("Existing collection deleted.")
    else:
        print("No existing collection found.")

    print("Database reset complete.")


if __name__ == "__main__":
    reset_database()
