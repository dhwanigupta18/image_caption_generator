# 📸 Image-to-Caption Generator

A Flask web application that generates natural language captions for uploaded images using a combination of the BLIP vision-language model and OpenAI's GPT for refinement.

## ✨ Features

* **Image Upload:** Easily upload images through a simple web interface.
* **BLIP Captioning:** Generates an initial descriptive caption using the powerful BLIP model.
* **Local Development Server:** Built with Flask for easy local deployment.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.8+
* Git
* An OpenAI API Key (for GPT refinement)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/image-caption-generator.git](https://github.com/your-username/image-caption-generator.git)
    cd image-caption-generator
    ```
    (Replace `your-username` with your actual GitHub username)

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    Create a file named `.env` in the root of your project directory and add your API key:
    ```
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
    ```
    **Important:** Replace `YOUR_OPENAI_API_KEY_HERE` with your actual key. This file is ignored by Git for security.

### Running the Application

1.  **Activate your virtual environment** (if not already active):
    ```bash
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

2.  **Run the Flask application:**
    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to `http://127.0.0.1:5000/`.

## 🛠️ Technologies Used

* **Python**
* **Flask:** Web framework
* **Hugging Face Transformers:** For BLIP model (`Salesforce/blip-image-captioning-base`)
* **PyTorch:** Deep learning framework
* **Pillow:** Image processing
* **OpenAI API:** For GPT-3.5 Turbo (or other GPT models)
* **python-dotenv:** For environment variable management

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, feel free to open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (You might add this file later if you want a specific license).

## 📞 Contact

[Dhwani Gupta/[(https://github.com/dhwanigupta18)] - (Optional: dhwanigupta18@gmail.com)