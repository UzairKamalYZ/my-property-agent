from langchain_core.documents import Document


def listings_to_documents(listings: list[dict]) -> list[Document]:
    """
    Convert property listings into LangChain Documents
    suitable for vector embedding and similarity search.
    """

    documents: list[Document] = []

    for listing in listings:
        # Defensive defaults
        property_type = listing.get("type", "Unknown property")
        city = listing.get("city", "Unknown city")
        rent = listing.get("rent", "Unknown rent")
        description = listing.get("description", "")
        url = listing.get("url", "")

        text = (
            f"{property_type} in {city}. "
            f"Monthly rent {rent} EUR. "
            f"{description}"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "type": property_type,
                    "city": city,
                    "rent": rent,
                    "description": description,
                    "url": url,
                }
            )
        )

    return documents
