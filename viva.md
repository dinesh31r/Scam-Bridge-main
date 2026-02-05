1. What problem does Scam-Bridge Analytica solve?
Answer: It provides political sentiment analysis and a Q&A chatbot with optional RAG to answer questions using a local knowledge base and, if enabled, web search.

2. What are the two sentiment engines supported?
Answer: TextBlob (lexicon-based polarity) and a trained Logistic Regression model with TF-IDF features.

3. What does the TextBlob engine return?
Answer: A sentiment label (Positive/Negative/Neutral) and a polarity score in the range [-1, 1].

4. What does the Logistic Regression engine return?
Answer: A sentiment label and a confidence score in the range [0, 1] derived from predict_proba when available.

5. Where are the Logistic Regression artifacts stored?
Answer: `models/final_model.pkl` and `models/vectorizer.pkl`.

6. What happens if the LR artifacts are missing?
Answer: The app shows a warning and prevents LR analysis until the artifacts are available.

7. What is RAG in this project?
Answer: Retrieval-Augmented Generation that injects relevant snippets from local knowledge base files into the chatbot prompt.

8. Where are knowledge base files stored?
Answer: In `data/knowledge_base` with `.txt` or `.md` extensions.

9. How does the app build the local KB index?
Answer: It vectorizes KB text with TF‑IDF and stores the matrix to retrieve top‑K similar documents.

10. How are relevant KB sources selected?
Answer: By cosine similarity between the query vector and document vectors, selecting the top‑K matches.

11. How are the KB sources shown to the user?
Answer: The filenames are displayed under the chatbot response.

12. What is the Groq client used for?
Answer: It sends chat completion requests to the Groq LLM for responses.

13. How is the Groq API key provided?
Answer: Via `GROQ_API_KEY` in `.env` or an optional sidebar override for the current session.

14. What happens if no Groq API key is found?
Answer: The chatbot returns “Groq chatbot disabled (API key not found).”

15. What is Tavily used for?
Answer: Optional web search to retrieve up‑to‑date context when “Use Web Search” is enabled.

16. How do you enable Tavily in the app?
Answer: Add `TAVILY_API_KEY` to `.env`, install `tavily-python`, and toggle “Use Web Search (Tavily)”.

17. Does the chatbot always use RAG?
Answer: No, it is controlled by a sidebar toggle.

18. How is caching implemented?
Answer: A session cache keyed by query, RAG flag, and top‑K (plus web settings) to reuse responses.

19. Where is chat history stored?
Answer: In `data/chat_history.db` using SQLite.

20. How many chat messages are retained?
Answer: The latest 200 entries are kept.

21. What fields are saved for each chat?
Answer: Timestamp, user message, bot response, sentiment label, score, and sources.

22. What is the rate limiter doing?
Answer: It limits chatbot requests per minute based on a sidebar setting.

23. What is the purpose of the “Use Cache” toggle?
Answer: To avoid repeated LLM calls for identical queries/settings.

24. How is sentiment computed for the chatbot queries?
Answer: The same sentiment engine selected in the UI is applied to user queries.

25. What is the default LLM model used?
Answer: `llama-3.1-8b-instant` via Groq.

26. How does the app handle LLM failures?
Answer: It catches exceptions and returns “Groq LLM temporarily unavailable.”

27. Which UI framework is used?
Answer: Streamlit.

28. How is the UI themed?
Answer: Custom CSS injected via `st.markdown` with a glassy, light theme and modern typography.

29. What are the main sections of the app?
Answer: Text sentiment analysis, bulk CSV sentiment analysis, chatbot, and model performance.

30. Why does the CSV output show fewer rows sometimes?
Answer: It originally showed a preview; now it shows the full table with scroll.

31. What is the purpose of the “Summary” section in CSV analysis?
Answer: It shows rows analyzed, average score, and a sentiment distribution chart.

32. What chart is used for sentiment distribution?
Answer: A bar chart generated with `st.bar_chart`.

33. How is data cleaning performed for LR?
Answer: Using `clean_and_stem` in `functions/preprocess.py`.

34. What is the role of TF‑IDF in LR?
Answer: It converts text into numeric features for the Logistic Regression model.

35. What does the “Clear Chat History” button do?
Answer: It wipes chat records from SQLite and clears session history.

36. Can the app work without web access?
Answer: Yes, it operates with local KB and Groq LLM only.

37. What is the effect of turning off RAG and web search?
Answer: The chatbot answers from the LLM without any retrieved context.

38. How is `.env` loaded?
Answer: `python-dotenv` loads `.env` at app startup.

39. Why use `.env` instead of exporting keys each time?
Answer: It allows persistent configuration without setting shell variables.

40. How is the “top‑K sources” setting used?
Answer: It controls how many KB documents are retrieved and inserted into the prompt.

41. What is shown in the sidebar?
Answer: Branding, settings, chat controls, and optional debug information.

42. What does “Show Debug Info” display?
Answer: Whether keys are found, history length, and RAG/web settings.

43. What happens if the KB folder is empty?
Answer: The app shows an info message indicating that RAG sources are missing.

44. How does the app ensure UI readability in inputs?
Answer: It sets explicit text color and caret color for input elements.

45. What is the output of the bulk CSV analysis?
Answer: A downloadable CSV with sentiment labels and scores added.

46. Why is `st.cache_data` used?
Answer: To avoid recomputing KB document loading and indexing on every run.

47. Why is `st.cache_resource` used for artifacts?
Answer: To load models once and reuse them across reruns.

48. How are chatbot responses stored in cache?
Answer: In a session dictionary keyed by query and settings.

49. What is the difference between RAG sources and web sources?
Answer: RAG sources are local KB files, web sources come from Tavily search results.

50. How would you deploy this app?
Answer: Host the Streamlit app on a server, set `.env` keys, and run `streamlit run app.py`.

51. Why did you choose Streamlit instead of a traditional web framework?
Answer: Streamlit enables rapid prototyping of data apps with minimal boilerplate and built-in UI components.

52. What are the limitations of your current approach?
Answer: It relies on external APIs for LLM/web search, is limited by model context, and may be slow on very large CSVs.

53. How would you handle very large CSV files efficiently?
Answer: Use chunked processing, background jobs, and caching; avoid loading all rows into memory at once.

54. How did you validate the sentiment model performance?
Answer: By evaluating accuracy, precision, recall, and F1 on a held‑out test set.

55. What could cause bias in sentiment predictions?
Answer: Training data imbalance, domain shift, and lexicon limitations in TextBlob.

56. How would you improve chatbot answer reliability?
Answer: Add stricter grounding with citations, better retrieval, and refuse to answer when sources are missing.

57. How do you ensure user privacy?
Answer: Avoid storing sensitive data, keep chat history local, and protect API keys via `.env`.

58. What’s the difference between RAG and web search here?
Answer: RAG uses curated local documents; web search uses live external sources.

59. Why did you use TF‑IDF instead of embeddings?
Answer: TF‑IDF is lightweight, fast, and sufficient for small local KBs.

60. What trade-offs exist between TextBlob and Logistic Regression?
Answer: TextBlob is simple and explainable but less accurate; LR is more accurate but needs training artifacts.

61. How would you support multiple languages?
Answer: Use language detection, multilingual embeddings, and per-language sentiment models or lexicons.

62. How would you secure the API keys in production?
Answer: Use environment variables and secret managers, not checked into Git.

63. How do you handle rate limits from external APIs?
Answer: Throttle requests, add retries with backoff, and cache responses.

64. What happens if a web search result is incorrect?
Answer: The answer can be wrong; mitigation includes source ranking, verification, and user-visible citations.

65. How would you make the UI more accessible?
Answer: Improve contrast, add aria labels, and ensure keyboard navigation works.

66. Why did you store chat history in SQLite?
Answer: It is lightweight, file‑based, and easy to use for small apps.

67. How would you scale this app to many users?
Answer: Add a backend API, move storage to a database, and deploy behind a load balancer.

68. How do you test this system?
Answer: Unit test preprocessing, mock LLM and search calls, and run UI smoke tests.

69. What are possible failure points in the pipeline?
Answer: Missing API keys, unavailable LLM/search services, or corrupted model artifacts.

70. If you had more time, what would you improve?
Answer: Add stronger retrieval, better caching, user auth, and more robust monitoring.
