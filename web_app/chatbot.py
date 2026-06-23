"""
AI Chatbot for Plant Disease Information
Supports Groq and Google Gemini APIs
"""

import os
import re
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PlantDiseaseChat:
    def __init__(self, api_key=None, provider='groq'):
        self.provider = provider.lower()

        # Load API key
        if api_key:
            self.api_key = api_key
        elif self.provider == 'groq':
            self.api_key = os.getenv("GROQ_API_KEY")
        else:
            self.api_key = os.getenv("GEMINI_API_KEY")

        self.client = None
        self.model = None

        self.temperature = 0.7
        self.max_tokens = 1024

        if self.api_key:
            self._initialize_client()

    # ---------------------------------------------------
    # Initialize API Client
    # ---------------------------------------------------
    def _initialize_client(self):
        try:
            if self.provider == 'groq':
                from groq import Groq
                self.client = Groq(api_key=self.api_key)

                # ✅ UPDATED MODEL (FIXED)
                self.model = "llama-3.1-8b-instant"

            elif self.provider == 'gemini':
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel("gemini-2.5-flash")

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except ImportError as e:
            raise ImportError(f"Missing dependency: {str(e)}")

    # ---------------------------------------------------
    # Clean AI Response
    # ---------------------------------------------------
    def _clean_response(self, text):
        if not text:
            return text

        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\^\\circ', '°', text)
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\$([^$]+)\$', r'\1', text)

        return text.strip()

    # ---------------------------------------------------
    # Main Chat Function
    # ---------------------------------------------------
    def get_response(self, user_message, disease_name=''):
        if not self.client:
            return "⚠️ API key not configured. Please add it in .env file."

        system_prompt = f"""
You are an expert agricultural assistant.

Detected disease: {disease_name if disease_name else 'Not specified'}

Provide:
1. Disease description
2. Symptoms
3. Treatment
4. Prevention
5. When to consult expert

Keep answers simple, practical, and farmer-friendly.
Avoid LaTeX or special formatting.
"""

        try:
            if self.provider == 'groq':
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return self._clean_response(response.choices[0].message.content)

            elif self.provider == 'gemini':
                prompt = f"{system_prompt}\nUser: {user_message}"
                response = self.client.generate_content(prompt)
                return self._clean_response(response.text)

        except Exception as e:
            return f"Error: {str(e)}"

    # ---------------------------------------------------
    # Disease Info Helper
    # ---------------------------------------------------
    def get_disease_info(self, disease_name):
        prompt = f"""
Explain plant disease: {disease_name}

Include:
- Cause
- Symptoms
- Treatment
- Prevention
"""
        return self.get_response(prompt, disease_name)

    # ---------------------------------------------------
    # Get JSON Cards for UI
    # ---------------------------------------------------
    def get_disease_cards(self, top_3_predictions):
        if not self.client:
            return {"error": "API key not configured."}
            
        prompt = f"""
You are an expert agricultural assistant helping farmers identify and treat plant diseases.

The system has predicted the following top diseases for an uploaded leaf image:
{top_3_predictions}

Your task is to generate a structured response for a CARD-BASED USER INTERFACE.

IMPORTANT:
- Generate MULTIPLE CARDS (one per disease)
- Each card must be detailed but easy to understand
- Use simple, farmer-friendly language
- Avoid scientific jargon unless necessary
- Keep content practical and actionable

Each CARD must include:

1. disease_name
2. confidence (percentage)
3. severity (Mild / Moderate / Severe)
4. short_summary (1–2 lines explaining the disease)
5. symptoms (3–5 bullet points)
6. causes (2–3 simple lines)
7. treatment (step-by-step bullet points)
8. prevention (bullet points)
9. risk_level (Low / Medium / High)
10. recommended_action (very short immediate advice)
11. language (default: English)

UI/UX INSTRUCTIONS:
- The frontend will display ONE CARD at a time
- Users can navigate using NEXT and PREVIOUS buttons
- So make each card COMPLETE and INDEPENDENT
- Do NOT reference "above" or "below"
- Do NOT assume multiple cards are visible together

MULTILINGUAL SUPPORT:
- Keep sentences simple so they can be translated easily
- Avoid idioms or complex grammar

OUTPUT FORMAT (STRICT JSON):
{{
  "cards": [
    {{
      "disease_name": "",
      "confidence": "",
      "severity": "",
      "short_summary": "",
      "symptoms": [],
      "causes": "",
      "treatment": [],
      "prevention": [],
      "risk_level": "",
      "recommended_action": "",
      "language": "English"
    }}
  ]
}}
"""
        try:
            if self.provider == 'groq':
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a JSON outputting expert agricultural AI. Always respond in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=4096,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
                return json.loads(result_text)

            elif self.provider == 'gemini':
                sys_prompt = "You are a JSON outputting expert agricultural AI. Always respond in valid JSON format without markdown code blocks."
                full_prompt = f"{sys_prompt}\nUser: {prompt}"
                response = self.client.generate_content(full_prompt)

                # Robustly extract JSON — handles ```json ... ``` wrappers and stray text
                raw = response.text.strip()
                # Try to extract JSON block if wrapped in markdown
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', raw, re.DOTALL)
                clean_json = json_match.group(1) if json_match else raw
                # Strip any remaining markdown fences
                clean_json = re.sub(r'^```(?:json)?\s*|\s*```$', '', clean_json).strip()
                return json.loads(clean_json)

        except Exception as e:
            return {"error": str(e)}