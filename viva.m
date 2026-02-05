Q1. What problem does Scam-Bridge Analytica solve?
A1. It provides political sentiment analysis and a Q&A chatbot with optional RAG to answer questions using a local knowledge base and, if enabled, web search.

Q2. What are the two sentiment engines supported?
A2. TextBlob (lexicon-based polarity) and a trained Logistic Regression model with TF-IDF features.

Q3. What does the TextBlob engine return?
A3. A sentiment label (Positive/Negative/Neutral) and a polarity score in the range [-1, 1].

Q4. What does the Logistic Regression engine return?
A4. A sentiment label and a confidence score in the range [0, 1] derived from predict_proba when available.

Q5. Where are the Logistic Regression artifacts stored?
A5. `models/final_model.pkl` and `models/vectorizer.pkl`.

Q6. What happens if the LR artifacts are missing?
A6. The app shows a warning and prevents LR analysis until the artifacts are available.

Q7. What is RAG in this project?
A7. Retrieval-Augmented Generation that injects relevant snippets from local knowledge base files into the chatbot prompt.

Q8. Where are knowledge base files stored?
A8. In `data/knowledge_base` with `.txt` or `.md` extensions.

Q9. How does the app build the local KB index?
A9. It vectorizes KB text with TF‑IDF and stores the matrix to retrieve top-K similar documents.

Q10. How are relevant KB sources selected?
A10. By cosine similarity between the query vector and document vectors, selecting the top-K matches.

Q11. How are the KB sources shown to the user?
A11. The filenames are displayed under the chatbot response.

Q12. What is the Groq client used for?
A12. It sends chat completion requests to the Groq LLM for responses.

Q13. How is the Groq API key provided?
A13. Via `GROQ_API_KEY` in `.env` or an optional sidebar override for the current session.

Q14. What happens if no Groq API key is found?
A14. The chatbot returns “Groq chatbot disabled (API key not found).”

Q15. What is Tavily used for?
A15. Optional web search to retrieve up-to-date context when “Use Web Search” is enabled.

Q16. How do you enable Tavily in the app?
A16. Add `TAVILY_API_KEY` to `.env`, install `tavily-python`, and toggle “Use Web Search (Tavily)”.

Q17. Does the chatbot always use RAG?
A17. No, it is controlled by a sidebar toggle.

Q18. How is caching implemented?
A18. A session cache keyed by query, RAG flag, and top‑K (plus web settings) to reuse responses.

Q19. Where is chat history stored?
A19. In `data/chat_history.db` using SQLite.

Q20. How many chat messages are retained?
A20. The latest 200 entries are kept.

Q21. What fields are saved for each chat?
A21. Timestamp, user message, bot response, sentiment label, score, and sources.

Q22. What is the rate limiter doing?
A22. It limits chatbot requests per minute based on a sidebar setting.

Q23. What is the purpose of the “Use Cache” toggle?
A23. To avoid repeated LLM calls for identical queries/settings.

Q24. How is sentiment computed for the chatbot queries?
A24. The same sentiment engine selected in the UI is applied to user queries.

Q25. What is the default LLM model used?
A25. `llama-3.1-8b-instant` via Groq.

Q26. How does the app handle LLM failures?
A26. It catches exceptions and returns “Groq LLM temporarily unavailable.”

Q27. Which UI framework is used?
A27. Streamlit.

Q28. How is the UI themed?
A28. Custom CSS injected via `st.markdown` with a glassy, light theme and modern typography.

Q29. What are the main sections of the app?
A29. Text sentiment analysis, bulk CSV sentiment analysis, chatbot, and model performance.

Q30. Why does the CSV output show fewer rows sometimes?
A30. Previously it showed a head preview; now it shows the full table with scroll.

Q31. What is the purpose of the “Summary” section in CSV analysis?
A31. It shows rows analyzed, average score, and a sentiment distribution chart.

Q32. What chart is used for sentiment distribution?
A32. A bar chart generated with `st.bar_chart`.

Q33. How is data cleaning performed for LR?
A33. Using `clean_and_stem` in `functions/preprocess.py`.

Q34. What is the role of TF‑IDF in LR?
A34. It converts text into numeric features for the Logistic Regression model.

Q35. What does the “Clear Chat History” button do?
A35. It wipes chat records from SQLite and clears session history.

Q36. Can the app work without web access?
A36. Yes, it operates with local KB and Groq LLM only.

Q37. What is the effect of turning off RAG and web search?
A37. The chatbot answers from the LLM without any retrieved context.

Q38. How is `.env` loaded?
A38. `python-dotenv` loads `.env` at app startup.

Q39. Why use `.env` instead of exporting keys each time?
A39. It allows persistent configuration without setting shell variables.

Q40. How is the “top‑K sources” setting used?
A40. It controls how many KB documents are retrieved and inserted into the prompt.

Q41. What is shown in the sidebar?
A41. Branding, settings, chat controls, and optional debug information.

Q42. What does “Show Debug Info” display?
A42. Whether keys are found, history length, and RAG/web settings.

Q43. What happens if the KB folder is empty?
A43. The app shows an info message indicating that RAG sources are missing.

Q44. How does the app ensure UI readability in inputs?
A44. It sets explicit text color and caret color for input elements.

Q45. What is the output of the bulk CSV analysis?
A45. A downloadable CSV with sentiment labels and scores added.

Q46. Why is `st.cache_data` used?
A46. To avoid recomputing KB document loading and indexing on every run.

Q47. Why is `st.cache_resource` used for artifacts?
A47. To load models once and reuse them across reruns.

Q48. How are chatbot responses stored in cache?
A48. In a session dictionary keyed by query and settings.

Q49. What is the difference between RAG sources and web sources?
A49. RAG sources are local KB files, web sources come from Tavily search results.

Q50. How would you deploy this app?
A50. By hosting the Streamlit app on a server, setting `.env` keys, and running `streamlit run app.py`.
