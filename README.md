# 🌾 Farmer AI Assistant

**An AI-powered agricultural assistant that combines plant disease detection with Retrieval-Augmented Generation (RAG) to provide understandable, context-based farming guidance.**

The Farmer AI Assistant helps users identify plant diseases from leaf images and obtain relevant agricultural information, including symptoms, possible causes, prevention, and management recommendations.

## 📌 Project Overview

Farmers may face difficulties identifying crop diseases and accessing understandable, relevant agricultural information. This project combines computer vision and generative AI to make agricultural guidance more accessible.

The system integrates:
- **Plant Disease Detection:** A deep learning model based on MobileNetV2 classifies plant leaf images.
- **Agricultural Question Answering:** A RAG pipeline retrieves relevant information from a local agricultural knowledge base.
- **AI-Generated Advice:** Google's Gemini model generates farmer-friendly responses using the retrieved agricultural context.
- **Interactive Web Interface:** Streamlit provides an interface for interacting with the assistant.

## ✨ Key Features

- 🌱 **Plant Disease Classification** — Predicts a disease or healthy-leaf category from an uploaded image.
- 🔎 **Knowledge Retrieval** — Uses semantic embeddings and FAISS similarity search to retrieve relevant agricultural documents.
- 🤖 **Generative AI Responses** — Uses Gemini to generate contextual agricultural explanations.
- 📚 **Agricultural Knowledge Base** — Retrieves information from locally stored agricultural text documents.
- 💬 **Interactive Chat Interface** — Allows users to submit agricultural questions and view responses.
- 🛡️ **Practical Farming Guidance** — Supports explanations of disease symptoms, causes, prevention, and management when relevant information is available.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web interface | Streamlit |
| Deep learning | TensorFlow / Keras |
| Image classification | MobileNetV2 |
| Generative AI | Google Gemini |
| LLM integration | LangChain |
| Text embeddings | FastEmbed |
| Vector similarity search | FAISS |
| Image processing | Pillow |
| Environment configuration | python-dotenv |

## 🧠 System Architecture

The application combines two processing pipelines.

### 1. Plant Disease Detection Pipeline

```text
Upload Plant Leaf Image
          ↓
Image Preprocessing
          ↓
MobileNetV2 Classification Model
          ↓
Predicted Disease / Healthy Class
          ↓
Display Prediction
```

### 2. RAG-Based Agricultural Question Answering

```text
Agricultural Question
          ↓
Sentence Embedding
          ↓
FAISS Similarity Search
          ↓
Retrieve Relevant Documents
          ↓
Retrieved Context + User Question
          ↓
Google Gemini
          ↓
Farmer-Friendly Response
```

### Integrated Workflow

The predicted disease can be used as the subject of an agricultural query. The RAG pipeline retrieves relevant information from the knowledge base, and the language model generates an explanation based on the available context.

The quality of the response depends on the relevance and completeness of the retrieved documents.

## 📊 Plant Disease Classification Model

The computer vision component uses transfer learning with MobileNetV2.

**Dataset:** PlantVillage

**Dataset characteristics:**
- Approximately 54,000 leaf images in the dataset version used.
- 38 plant disease and healthy-leaf classes.
- Separate training, validation, and test datasets.

**Image input size:** 224 × 224 pixels

**Batch size:** 32

**Model approach:** MobileNetV2 with transfer learning.

### Evaluation

The classifier can be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The test-set accuracy observed during development was approximately **95.72%**. This is an experimental result on the selected test dataset and does not guarantee equivalent performance on real-world farm images.

## 📂 Project Structure

```text
Farmer-Chatbot/
│
├── app.py                  # Streamlit application
├── data/
│   └── agriculture.txt     # Agricultural knowledge base
├── .env                    # API key configuration (not committed)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── models/
    └── plant_disease_model.keras
                             # Trained classifier, if stored locally
```

*Note: The structure above is illustrative. Adjust filenames and folders to match the actual repository.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- Git
- A Google Gemini API key
- Sufficient memory and storage for the model and dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/srimayadam-png/Farmer-Chatbot.git
cd Farmer-Chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key. Do not commit the `.env` file or publish your API key.

### 5. Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, typically:

```text
http://localhost:8501
```

**Important:** Ensure that the trained disease classification model and agricultural knowledge-base files are present at the paths expected by the application.

## 🚀 How to Use

1. Start the Streamlit application.
2. Upload a supported plant leaf image if using the disease detection component.
3. Review the predicted disease or healthy-leaf category.
4. Ask an agricultural question related to the crop or predicted disease.
5. Review the generated response and retrieved agricultural information.
6. Use the output as preliminary guidance and verify important treatment decisions with a qualified agricultural expert.

## 🔐 API Key and Security

- Store API credentials in environment variables.
- Add `.env` and other secret files to `.gitignore`.
- Never expose API keys in source code, screenshots, logs, or public commits.
- Monitor Gemini API quotas and rate limits.
- Avoid uploading sensitive information to external AI services.

## ⚠️ Limitations

- Disease classification is limited to the classes represented in the training dataset.
- Images captured under real farming conditions may differ from the dataset images.
- Similar-looking diseases may be confused by the classifier.
- Agricultural answers depend on the information available in the knowledge base and the quality of retrieval.
- Generative AI may produce incomplete or inaccurate recommendations.
- API quota limits, network connectivity, and model availability can affect response generation.
- The system does not replace laboratory diagnosis, field inspection, or professional agricultural advice.

## 🔮 Future Enhancements

- Support for additional crops and disease categories.
- Regional-language support for Indian farmers.
- Voice-based question answering.
- Expansion of the knowledge base with verified agricultural publications.
- Improved retrieval evaluation and source attribution.
- Testing with field images collected under diverse lighting and environmental conditions.
- Deployment as a publicly accessible web application.

## 🌍 Social Impact

The project aims to improve access to agricultural information by combining image-based disease classification with natural-language explanations. It supports the broader goal of promoting informed farming decisions and contributes to the objectives of **United Nations Sustainable Development Goal 2: Zero Hunger**.

## 👩‍💻 Contributing

Contributions, suggestions, and bug reports are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Submit a pull request describing the changes.

## 📄 License

A license has not been specified here. Add a suitable open-source license to the repository before explicitly permitting reuse or redistribution.

## 🙏 Acknowledgements

- PlantVillage dataset for plant disease image data.
- TensorFlow and Keras for deep learning tools.
- Streamlit for the interactive application interface.
- FAISS and FastEmbed for semantic retrieval.
- Google Gemini for generative AI capabilities.
- LangChain for integrating language-model components.

---

**Farmer AI Assistant — Combining Computer Vision and Generative AI for Accessible Agricultural Guidance.**
