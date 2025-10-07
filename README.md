# AI Emotion Journal

AI Emotion Journal is a web-based journaling application that leverages AI to analyze and track the emotional content of user journal entries. The app provides a simple interface for users to create, view, and manage their personal journal entries, with emotion analysis powered by a machine learning model.

## Features

- **User Authentication:** Secure sign-in and account creation for personalized journaling.
- **Create Journal Entries:** Users can write and submit new entries through a user-friendly form.
- **Emotion Analysis:** Each entry is analyzed by an AI model to detect and display the predominant emotion.
- **View Past Entries:** Users can browse previous entries, view their content, associated emotion, and date.
- **Database Storage:** Entries and user data are stored in a backend database for persistence.

## Technology Stack

- **Frontend:** React.js (JavaScript)
- **Backend:** Flask (Python)
- **Database:** MySQL
- **AI Model:** PyTorch-based emotion classification

## Folder Structure

- `journal/` - Main application folder
  - `src/` - React frontend source code
  - `database/` - Database schema and Python DB logic
  - `public/` - Static assets for the frontend
- `model/` - AI model files and emotion analysis scripts
  - `tokenizer/` - Tokenizer configuration for the model

## Getting Started

1. **Install Dependencies**
   - Frontend: Navigate to `journal/src` and run `npm install`.
   - Backend: Set up a Python virtual environment and install required packages (see `requirements.txt` if available).

2. **Run the Application**
   - Start the backend Flask server (usually on port 5000).
   - Start the frontend React app with `npm start`.

3. **Access the App**
   - Open your browser and go to `http://localhost:3000` for the frontend.
   - The backend API runs at `http://localhost:5000`.

## Usage

- **Sign In / Sign Up:** Create an account or log in to access your journal.
- **Create Entry:** Write a new journal entry and submit. The AI will analyze the emotion.
- **View Entries:** Browse your previous entries, see the detected emotion and entry date.

## AI Emotion Analysis

The application uses a PyTorch model to classify the emotion of each journal entry. The model and tokenizer files are located in the `model/` directory.

