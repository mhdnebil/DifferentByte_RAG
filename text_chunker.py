def split_text(text, chunk_size=300, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    from pdf_reader import extract_text

    text = extract_text("DifferentByte Company Information.pdf")

    chunks = split_text(text)

    print("\nTotal Chunks:", len(chunks))

    for index, chunk in enumerate(chunks):

        print(f"\n--- CHUNK {index + 1} ---")

        print(chunk)