# AI Model Comparison Chat Interface

A clean, interactive web application that allows users to compare responses from different AI models (Google's Gemini Pro and OpenAI's GPT) side by side. This project demonstrates how to integrate multiple AI models into a single interface for direct comparison and learning purposes.

## Features

- 🤖 Real-time interaction with multiple AI models
- 🔄 Easy switching between Gemini and OpenAI models
- 📊 Proper formatting of structured data (tables, lists)
- 💾 Save chat history functionality
- 📱 Responsive design for mobile and desktop
- 🎨 Clean, modern user interface

## Live Demo

[Link to your deployed app will go here]

## Prerequisites

- Python 3.12.9 or higher
- Google AI API key
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-model-comparison.git
cd ai-model-comparison
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.template` to `.env`
   - Add your API keys:
     ```
     GOOGLE_API_KEY=your_gemini_api_key
     OPENAI_API_KEY=your_openai_api_key
     ```

## Running Locally

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
ai-model-comparison/
├── app.py                 # Main FastAPI application
├── static/
│   └── style.css         # CSS styles
├── templates/
│   └── index.html        # HTML template
├── .env                  # Environment variables (not in repo)
├── .env.template         # Template for environment variables
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## API Endpoints

- `GET /`: Main chat interface
- `POST /chat/gemini`: Endpoint for Gemini model interactions
- `POST /chat/openai`: Endpoint for OpenAI model interactions

## Deployment

This application can be deployed to various platforms:

- **Render**: [Deployment Guide](link-to-render-guide)
- **Streamlit Cloud**: [Deployment Guide](link-to-streamlit-guide)
- **Heroku**: [Deployment Guide](link-to-heroku-guide)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google AI for Gemini Pro API
- OpenAI for GPT API
- FastAPI framework
- Community contributors

## Contact

Developed by [Lindsay Hiebert](https://www.linkedin.com/in/lindsayhiebert/)
- GitHub: [lhiebert01](https://github.com/lhiebert01)
- LinkedIn: [lindsayhiebert](https://www.linkedin.com/in/lindsayhiebert/)