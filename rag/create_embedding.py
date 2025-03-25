from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import pdfplumber

PROJECT_ID = "walmart-retail-media"
MODEL_ID = "text-embedding-004"


class ExtractCreateEmbeddings:

    def __init__(self, file_path):
        self.model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.path = file_path
        print("PAth:::   ", self.path)


    def embed_text(self, text: list):
        embeddings = self.model.get_embeddings([text])
        res = []
        for embedding in embeddings:
            res.append(embedding.values)
            # print(len(embedding.values))
        # print(len(res))
        return res
   

    # Function to extract text from PDF
    def extract_text_from_pdf(self):
        # Open the PDF file
        with pdfplumber.open(self.path) as pdf:
            all_text = ""
            extract = []
            # Iterate over all the pages
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()  # Extract text from each page
                if text:
                    embed = self.embed_text(text)
                    # extract.append([text,embed[0]])
                    # extract[f"page_{i+1}"] = [text, embed[0]]
                    extract.append([text, embed[0]])
                print(f"Extracted text from page {i + 1}")

            return extract
        
def embed_texts(text: list):
        print("Model load")
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        print("Embedding")
        embeddings = model.get_embeddings([text])
        res = []
        print("start loop")
        for embedding in embeddings:
            res.append(embedding.values)
            # print(len(embedding.values))
        # print(len(res))
        return res

    


# print(extracted_text)


# with open("../data/India Employee Handbook - 2024.pdf", 'rb') as f:
#     print("Hooray")
#     f.close()

# with pdfplumber.open("../data/India Employee Handbook - 2024.pdf") as pdf:
#     for i, page in enumerate(pdf.pages):
#         print(page.extract_text())