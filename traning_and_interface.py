#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('writefile', 'my_pipeline.py', 'import re\nimport pandas as pd\nimport spacy\nimport nltk\nimport numpy as np\nfrom nltk.corpus import stopwords\nfrom sentence_transformers import SentenceTransformer\nfrom sklearn.preprocessing import StandardScaler\n\n# ===========================\n# Init NLP + tools\n# ===========================\nnltk.download(\'stopwords\', quiet=True)\n\nnlp = spacy.load("en_core_web_sm")\nnltk_stopwords = set(stopwords.words(\'english\'))\nspacy_stopwords = nlp.Defaults.stop_words\ncombined_stopwords = spacy_stopwords.union(nltk_stopwords)\n\nbert_model = SentenceTransformer(\'all-MiniLM-L6-v2\')\nscaler = StandardScaler()\n\n# Regex parser\nlog_pattern = re.compile(\n    r"####<(?P<timestamp>[^>]+)>\\s*"\n    r"<(?P<level>[^>]+)>\\s*"\n    r"<(?P<component>[^>]+)>\\s*"\n    r"<(?P<machine>[^>]+)>\\s*"\n    r"<(?P<server>[^>]+)>\\s*"\n    r"<(?P<thread>.*?)>\\s*"\n    r"<(?P<kernel_info>.*?)>\\s*"\n    r"<>\\s*<>\\s*"  # skipped empty fields\n    r"<(?P<epoch>\\d+)>\\s*"\n    r"<(?P<code>bea-\\d+)>\\s*"\n    r"<(?P<message>.*?)(?:>)?\\s*$"\n)\n\n# ===========================\n# Cleaning\n# ===========================\ndef clean_log_message(text):\n    if not isinstance(text, str):\n        return ""\n    text = re.sub(r\'[^a-zA-Z ]+\', \' \', text.lower())\n    doc = nlp(text)\n    return \' \'.join([\n        token.lemma_ for token in doc\n        if token.is_alpha and token.lemma_ not in combined_stopwords and len(token.lemma_) > 1\n    ])\n\n# ===========================\n# Row-by-row Preprocessing\n# ===========================\ndef preprocess_log_line(line):\n    """\n    Take one log line -> return dict with parsed + features.\n    """\n    line = line.strip().lower()\n    match = log_pattern.match(line)\n    if not match:\n        return None  # skip invalid line\n\n    log_dict = match.groupdict()\n    log_dict[\'cleaned_message\'] = clean_log_message(log_dict[\'message\'])\n\n    # BERT features\n    X_bert = bert_model.encode([log_dict[\'cleaned_message\']])[0]\n\n    # One-hot like encoding (simple manual, row by row)\n    level = f"level_{log_dict[\'level\']}"\n    component = f"component_{log_dict[\'component\']}"\n\n    # numeric: message length (scaled)\n    msg_len = len(log_dict[\'message\'])\n    msg_len_scaled = scaler.fit_transform(np.array(msg_len).reshape(-1,1))[0][0]\n\n    # final features vector\n    log_dict[\'features\'] = np.hstack((\n        X_bert,\n        [msg_len_scaled]\n    ))\n\n    return log_dict\nimport re\nimport pandas as pd\nfrom datetime import datetime\n\n# ---------------------------------\n# Log level categories\n# ---------------------------------\nlog_level_categories = {\n    \'emergency\': \'System Down\',\n    \'alert\': \'Security Breach Detected\',\n    \'critical\': \'Service Outage\',\n    \'error\': \'Operation Failed\',\n    \'warning\': \'Potential Issue\',\n    \'warnning\': \'Potential Issue\',   # typo case\n    \'notice\': \'Significant Event\',\n    \'info\': \'Normal Operation\',\n    \'debug\': \'Troubleshooting\',\n    \'trace\': \'Execution Flow\',\n    \'failure\': \'Critical System Failure\',\n    \'fatal\': \'Process Termination\',\n    \'j2ee_error\': \'Container Failure\',\n    \'deployment_failure\': \'Deployment Aborted\',\n    \'tx_rollback\': \'Transaction Failed\',\n    \'unavailable\': \'Admin Server Down\',\n    \'restarted\': \'Server Rebooted\',\n    \'recoverable\': \'Recoverable Error Occurred\',\n    \'suspicious\': \'Suspicious Activity Detected\',\n    \'emergency_shutdown\': \'Emergency System Shutdown\',\n    \'startup\': \'System Initialization\',\n    \'shutdown\': \'System Shutdown\',\n    \'maintenance\': \'Scheduled System Maintenance\',\n}\n\n# ---------------------------------\n# Component regex categories\n# ---------------------------------\ncomponent_categories = {\n    r\'\\bj2ee\\b\': \'j2ee container\',\n    r\'\\bejbs?\\b\': \'enterprise java beans\',\n    r\'\\bservlet\\b\': \'web component\',\n    r\'\\bjsp\\b\': \'presentation layer\',\n    r\'\\bjndi\\b\': \'naming services\',\n    r\'\\bjca\\b\': \'connector architecture\',\n    r\'weblogicserver\': \'core server\',\n    r\'security\': \'authentication & authorization\',\n    r\'deployer\': \'deployment engine\',\n    r\'jdbc\': \'database connectivity\',\n    r\'\\bjms\\b\': \'messaging services\',\n    r\'\\bcluster\\b\': \'server clustering\',\n    r\'console\': \'management interface\',\n    r\'nodemanager\': \'server instance controller\',\n    r\'workmanager\': \'thread pool management\',\n    r\'harvester\': \'metric collection\',\n    r\'stdout\': \'standard output logging\',\n    r\'appmerge\': \'application packaging\',\n    r\'diagnostics\': \'monitoring subsystem\',\n    r\'coherence\': \'in-memory data grid\',\n    r\'webapp\': \'web application framework\',\n    r\'wlst\': \'scripting tool\',\n    r\'snmp\': \'network management\',\n    r\'connector\': \'adapter integration\',\n    r\'jvm|jvm crash\': \'Java Virtual Machine\',\n    r\'datasource\': \'Database Connection Pool\',\n    r\'ssl|tls|keystore|truststore\': \'Secure Socket Layer\',\n    r\'webcontainer|servlet engine\': \'Web Container\',\n    r\'\\bdb\\b|sql|query|cursor\': \'Database Service\',\n    r\'weblogic server|server crash\': \'WebLogic Server\',\n    r\'clustering|splitbrain|partition\': \'Clustered Environment\',\n    r\'loadbalancer|balancer\': \'Load Balancer\',\n    r\'cache|coherence\': \'Memory Caching\',\n    r\'gc|garbage collection\': \'JVM Garbage Collection\',\n    r\'socket|port|connection refused|handshake\': \'Network Socket\',\n    r\'storage|store|filesystem|disk|i/o|nfs|volume|partition|quota\': \'Storage System\',\n    r\'memory|heap|outofmemory|oom\': \'Memory Issues\',\n    r\'overflow|buffer|exhausted\': \'Buffer Overflow\',\n    r\'connection|timeout|refused|reset\': \'Connection Issues\',\n    r\'crash|abort|failure|exception|error\': \'Critical Failures\',\n    r\'corruption|corrupted\': \'Data Corruption\',\n    r\'invalid|illegal|unsupported|unauthorized|violation\': \'Invalid Operation\',\n    r\'breach|intrusion|attack|malicious\': \'Security Breaches\',\n    r\'lock|constraint|deadlock\': \'Locking Issues\',\n    r\'disconnect|unreachable|reset\': \'Network Disconnects\',\n    r\'heartbeat|heartbeat failure|timeout\': \'Heartbeat Issues\',\n    r\'caching|cache|outofmemory|threshold\': \'Cache Issues\',\n    r\'redeployment|undeployment|deploy\': \'Deployment Issues\',\n    r\'rollback|commit|transaction\': \'Transaction Issues\',\n    r\'queue|consumer|producer|broker|topic\': \'Messaging Issues\',\n    r\'io|filesystem|disk|nfs|storage\': \'File System Issues\',\n    r\'authentication|authorization|session\': \'Authentication Issues\',\n}\n\n# ---------------------------------\n# Row-by-row enrichment\n# ---------------------------------\ndef enrich_log_row(row: dict) -> dict:\n    """\n    Take one parsed + cleaned log row (dict)\n    Return same row enriched with categories + derived features\n    """\n    enriched = dict(row)  # copy original\n\n    # ---- datetime ----\n    # ---- datetime ----\n    try:\n        epoch_val = enriched.get("epoch")\n        if epoch_val is not None:\n            # Cast to numeric before converting\n            epoch_val = pd.to_numeric(epoch_val, errors="coerce")\n        dt = pd.to_datetime(epoch_val, errors="coerce", unit="ms")\n    except Exception:\n        dt = pd.NaT\n    enriched["datetime"] = dt\n    enriched["timestamp"] = dt\n\n    # ---- categories ----\n    enriched["level_category"] = log_level_categories.get(\n        enriched.get("level", "").lower(), "Other"\n    )\n\n    comp = enriched.get("component", "")\n    enriched["component_category"] = next(\n        (v for k, v in component_categories.items() if re.search(k, str(comp), re.IGNORECASE)),\n        "Other Component"\n    )\n\n    # ---- derived features ----\n    msg = str(enriched.get("message", ""))\n    enriched["message_length"] = len(msg)\n    if pd.notna(dt):\n        enriched["hour"] = dt.hour\n        enriched["day_of_week"] = dt.day_name()\n    else:\n        enriched["hour"] = None\n        enriched["day_of_week"] = None\n\n    enriched["peak_hours"] = int(8 <= enriched["hour"] <= 18) if enriched["hour"] is not None else 0\n\n    # text feature (placeholder for vectorization)\n    enriched["text_feature"] = enriched.get("cleaned_message", "")\n\n    # (row-by-row ما عندناش rolling features كيما في batch)\n    enriched["time_diff_prev"] = 0\n    enriched["time_diff_server"] = 0\n    enriched["logs_15min_count"] = 1\n    enriched["warn_15min_count"] = 1 if enriched.get("level", "").lower() == "warning" else 0\n    enriched["msg_len_mean_10"] = enriched["message_length"]\n    enriched["msg_len_std_10"] = 0\n\n    return enriched\nfrom collections import Counter\nfrom functools import lru_cache\nimport pandas as pd\nimport requests\nimport os\nimport re\nimport json\n\n# Set your Groq API key (safely store this in env for production use)\nGROQ_API_KEY = os.getenv("GROQ_API_KEY", "API_KEY")\nGROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"\n\ndef call_groq(prompt: str):\n    """Call LLaMA 3 on Groq for zero-shot classification."""\n    headers = {\n        "Authorization": f"Bearer {GROQ_API_KEY}",\n        "Content-Type": "application/json"\n    }\n    payload = {\n        "model": GROQ_MODEL,\n        "messages": [\n            {"role": "system", "content": "You are a log classifier. Classify the input based on predefined labels."},\n            {"role": "user", "content": prompt}\n        ],\n        "temperature": 0.2,\n        "max_tokens": 512\n    }\n\n    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)\n    response.raise_for_status()\n    return response.json()["choices"][0]["message"]["content"].strip()\n\n\n# Candidate labels (your full dict here)\ncandidate_labels_by_level = {\n    "NOTICE": {\n        "connection closed": "Connection was closed, creating a new one",\n        "card list request": "Web service call to retrieve the card list",\n        "operation card listing": "Operation Card Listing executed with row count",\n        "sms webservice url call": "Accessed URL of SMS Express WebService alerts",\n        "expiration date logged": "Expiration date recorded or processed",\n        "multiple cards returned": "Card listing operation returned >1 rows",\n        "server starting": "WebLogic server is starting up",\n        "server started in development mode": "Server started in development mode",\n        "server started in production mode": "Server started in production mode",\n        "server shutdown initiated": "Shutdown of WebLogic server has started",\n        "shutdown class executed": "Shutdown class executed successfully",\n        "server state changed to running": "Server transitioned to RUNNING state",\n        "server state changed to standby": "Server transitioned to STANDBY state",\n        "server health changed to ok": "Server health marked as OK",\n        "machine reachable": "Machine reachable via Node Manager",\n        "node manager started": "Node Manager started and listening",\n        "cluster member added": "New member joined WebLogic cluster",\n        "cluster heartbeat received": "Heartbeat received from cluster member",\n        "jms server activated": "JMS server has been activated successfully",\n        "jms server paused": "JMS server paused temporarily",\n        "jms destination restarted": "JMS destination was restarted cleanly",\n        "jdbc driver registered": "JDBC driver registered without issues",\n        "datasource active": "JDBC datasource is active and accepting connections",\n        "diagnostic volume set": "Diagnostic volume level set (Low, Medium, High)",\n        "log file rotated": "Log rotation triggered and completed",\n        "audit log initialized": "Security audit log initialized successfully",\n        "security realm initialized": "Security realm configuration loaded",\n        "default keystore loaded": "Default identity keystore has been loaded",\n        "webapp initialized": "Web application successfully initialized",\n        "listener started": "Network listener started and bound to port",\n        "application update received": "Application update task started",\n        "deployment task completed": "Deployment task completed successfully",\n        "module prepared": "Module prepared for deployment",\n        "module activated": "Module successfully activated",\n        "library referenced": "Shared library successfully referenced",\n        "wlst command executed": "WLST command was executed without error",\n        "configuration changes saved": "Domain configuration saved via console/WLST",\n        "startup class executed": "Startup class has run without error",\n        "log broadcast received": "Log broadcast message received by admin server",\n    },\n\n    "INFO": {\n        "application deployment started": "Application deployment initiated on WebLogic",\n        "application deployed successfully": "Application deployed and active",\n        "jdbc driver loaded": "JDBC driver loaded and registered",\n        "jdbc datasource initialized": "Datasource pool initialized and ready",\n        "thread pool statistics": "Periodic thread pool statistics logged",\n        "garbage collection summary": "Garbage collection completed; summary logged",\n        "ssl certificate loaded": "SSL certificate loaded into keystore",\n        "session replication successful": "HTTP session replication completed without errors",\n        "jms queue threshold reached": "JMS queue depth reached configured threshold",\n        "cluster address resolved": "ClusterAddress DNS resolution successful",\n    },\n\n    "WARNING": {\n        # Standard WebLogic warnings\n        "stuck threads detected": "One or more threads marked STUCK in WebLogic",\n        "thread pool at capacity": "Thread pool requests queued; capacity reached",\n        "high heap usage": "Heap usage exceeded warning threshold",\n        "long gc pause": "GC pause time exceeded tuning target",\n        "datasource connection leak": "Potential JDBC connection leak detected",\n        "session replication failures": "Some sessions failed to replicate to secondary",\n        "jms destination paused": "JMS destination automatically paused due to overflow",\n        "low disk space": "File store or server volume low on disk space",\n        "ssl certificate expiring soon": "SSL certificate will expire soon; renewal advised",\n        "node manager unreachable": "Node Manager not responding to heartbeat",\n        "cluster member lost heartbeat": "Cluster member removed due to missed heartbeats",\n        "transaction rollback warning": "XA transaction may roll back due to timeout",\n        "unauthorized access attempt": "Login failed; invalid credentials or role",\n        "dns resolution warnings": "Intermittent DNS resolution failures detected",\n\n        # Custom / additional warnings\n        "application exception": "Application has thrown exception, unwinding now",\n        "datasource connection released": "Forcibly releasing inactive/harvested connection back into the data source connection pool",\n\n        # Extra (optional but frequent)\n        "server state change": "Server state changed to running",\n        "jdbc driver registered": "JDBC driver registered",\n    },\n\n    "ERROR": {\n        "server failed to start": "WebLogic server startup failed; see stack trace",\n        "deployment failed": "Application deployment failed on WebLogic server",\n        "out of memory errors": "JVM threw OutOfMemoryError",\n        "stuck thread time exceeded": "Stuck thread BEA-000337 exceeded MaxTime",\n        "jdbc connection failed": "Failed to obtain JDBC connection from datasource",\n        "connection pool exhausted": "All connections in pool are in use",\n        "transaction timeout": "JTA transaction timed out and rolled back",\n        "jms message loss": "Messages lost or undeliverable in JMS server",\n        "file store full": "Persistent store cannot write; disk full",\n        "ssl handshake failed": "SSL handshake exception; certificate or protocol mismatch",\n        "cluster communication failure": "Unable to send/receive multicast or unicast messages",\n        "node manager failed": "Node Manager encountered fatal error",\n        "authentication failures": "User authentication failed repeatedly",\n        "authorization failures": "User lacks required security role or policy",\n        "logging failures": "Unable to write to log file; I/O error",\n        "port binding errors": "Port already in use; server cannot listen",\n        "dns resolution failures": "DNS name could not be resolved",\n        "external service unavailability": "Dependent external service is down or unreachable",\n        "servlet loading issues": "Servlet failed to initialize correctly",\n        "jsp compilation errors": "JSP compilation failed during runtime",\n        "ejb lookup failures": "JNDI lookup for EJB failed",\n        "resource allocation failures": "Failed to allocate required system resources",\n        "health state failed": "Server health transitioned to FAILED",\n        "persistent store corruption": "Critical corruption in persistent store files",\n    },\n\n    "DEBUG": {\n        "debug security audit": "Verbose security audit logs enabled",\n        "debug jdbc driver": "Detailed JDBC driver debugging output",\n        "debug jms transport": "Low-level JMS transport debug messages",\n        "debug cluster messaging": "Cluster multicast & unicast debug logs",\n        "debug memory tracker": "Heap and native memory tracker debug output",\n        "debug jndi lookup": "Detailed JNDI lookup tracing",\n        "debug webservice invocation": "SOAP / REST client debug messages",\n        "debug transaction tracing": "Fine-grained JTA/XA transaction tracing",\n    },\n\n    "CRITICAL": {\n        "server panic": "WebLogic server issued panic and will exit",\n        "jvm crash": "HotSpot JVM crash detected; hs_err dump generated",\n        "heap dump generated": "Automatic heap dump due to OOM or panic",\n        "persistent store corruption": "Critical corruption in persistent store files",\n        "disk full": "Disk full; critical services cannot proceed",\n        "database unreachable": "Database unreachable for extended period",\n        "cluster master lost": "Cluster master failed; failover unsuccessful",\n        "license expired": "WebLogic license expired or invalid",\n        "fatal native memory leak": "Native memory leak exceeded safety threshold",\n        "fatal thread deadlock": "Fatal deadlock detected among JVM threads",\n        "security breach detected": "Potential security breach; immediate action required",\n        "unsatisfied link error": "Native library load failure caused fatal error",\n        "port binding failure critical": "Essential port binding failed; server exits",\n    }\n}\n\n\n\n@lru_cache(maxsize=None)\ndef cached_classify(text: str, level: str):\n    """Classify logs by restricting to candidate labels of given level first."""\n\n    level = (level or "").strip().upper()\n    label_dict = candidate_labels_by_level.get(level, {})\n\n    # Use only level-specific labels first\n    candidates = list(label_dict) if label_dict else [\n        label for group in candidate_labels_by_level.values() for label in group\n    ]\n\n    prompt = f"""You are given a log summary: "{text}"\n\nChoose the most appropriate label from the following list of labels:\n{\', \'.join(candidates)}\n\nReturn only the best-matching label from the list (use exact wording from the list)."""\n\n    best_label = call_groq(prompt).strip().lower()\n\n    # --- Matching strategy ---\n    for label in label_dict:\n        if best_label == label.lower():\n            return label, label_dict[label]\n    for label in label_dict:\n        if best_label in label.lower() or label.lower() in best_label:\n            return label, label_dict[label]\n    for level_group, group_dict in candidate_labels_by_level.items():\n        for label in group_dict:\n            if best_label == label.lower():\n                return label, group_dict[label]\n    for level_group, group_dict in candidate_labels_by_level.items():\n        for label in group_dict:\n            if best_label in label.lower() or label.lower() in best_label:\n                return label, group_dict[label]\n\n    # 4. No match → Ask LLM to generate class + description\n    message = (text or "").strip()\n    cleaned_message = re.sub(r\'[^\\w\\s]\', \' \', message)\n    cleaned_message = re.sub(r\'\\s+\', \' \', cleaned_message).strip().lower()\n\n    llm_prompt = f"""\nYou are an expert log classifier. A log summary could not be matched to predefined labels.\n\nOriginal text:\n"{message}"\n\nCleaned text:\n"{cleaned_message}"\n\nPlease propose:\n1) A concise label (2–5 words, lowercase, no punctuation) that could act as a new class.\n2) A one-sentence human-readable description of what the label means.\n\nReturn a JSON object ONLY, exactly like this:\n{{"label":"<label>", "description":"<one-sentence description>"}}\n"""\n\n    try:\n        llm_resp = call_groq(llm_prompt)\n    except Exception:\n        return best_label, "No matching description found."\n\n    label, description = "", ""\n    try:\n        parsed = json.loads(llm_resp)\n        if isinstance(parsed, dict):\n            label = parsed.get("label", "")\n            description = parsed.get("description", "")\n    except Exception:\n        m_label = re.search(r\'"label"\\s*:\\s*"([^"]+)"\', llm_resp)\n        m_desc = re.search(r\'"description"\\s*:\\s*"([^"]+)"\', llm_resp)\n        if m_label:\n            label = m_label.group(1)\n            description = m_desc.group(1) if m_desc else ""\n        else:\n            parts = [ln.strip() for ln in llm_resp.splitlines() if ln.strip()]\n            if parts:\n                label = parts[0]\n                description = " ".join(parts[1:]).strip()\n\n    if not label:\n        return best_label, "No matching description found."\n    return label.strip(), description.strip() or "No description generated."\n\n\n# --- Row-level classification ---\ndef classify_log_row(row):\n    """\n    Take a single preprocessed log row (dict or pandas.Series) and classify it.\n    Returns the row with added \'final_label\' and \'final_label_desc\'.\n    """\n    level = (row.get("level", "") or "").strip().upper()\n    context = " | ".join(filter(None, (row.get("message"), row.get("component"))))\n\n    label, desc = cached_classify(context, level)\n\n    row["final_label"] = label\n    row["final_label_desc"] = desc\n    return row\n\n\n# --- DataFrame-level classification ---\ndef get_cluster_level(df, cluster_id):\n    levels = df.loc[df["cluster"] == cluster_id, "level"]\n    return levels.mode().iat[0] if not levels.empty else ""\n\n\ndef top_keywords(df, cluster_id):\n    texts = df.loc[df["cluster"] == cluster_id, "cleaned_message"]\n    return Counter(" ".join(texts).split()).most_common(5)\n\n\ndef classify_logs(df):\n    cluster_label_map = {}\n    cluster_desc_map = {}\n\n    # Step 1: classify per cluster\n    for cl in df["cluster"].dropna().unique():\n        level = get_cluster_level(df, cl)\n        keywords = [k for k, _ in top_keywords(df, cl)]\n        label, desc = cached_classify(" ".join(keywords), level)\n        cluster_label_map[cl] = label\n        cluster_desc_map[cl] = desc\n\n    df["cluster_label"] = df["cluster"].map(cluster_label_map)\n    df["cluster_label_desc"] = df["cluster"].map(cluster_desc_map)\n\n    # Step 2: row-level classification if cluster-level missing\n    def classify_with_context(row):\n        level = row.get("level", "").strip().upper()\n        context = " | ".join(filter(None, (row.get("message"), row.get("component"))))\n        return pd.Series(cached_classify(context, level))\n\n    mask = df["cluster_label"].isna()\n    df.loc[mask, ["keyword_class", "keyword_class_desc"]] = df.loc[mask].apply(classify_with_context, axis=1)\n\n    # Step 3: Merge cluster and row-level classification\n    def merge_labels(row):\n        labels, descs = [], []\n        if pd.notna(row["cluster_label"]):\n            labels.append(row["cluster_label"])\n            descs.append(row["cluster_label_desc"])\n        if pd.notna(row.get("keyword_class")) and row["keyword_class"] not in labels:\n            labels.append(row["keyword_class"])\n            descs.append(row["keyword_class_desc"])\n        return pd.Series([" + ".join(labels), " | ".join(descs)])\n\n    df[["final_label", "final_label_desc"]] = df.apply(merge_labels, axis=1)\n    df.drop(columns=["keyword_class", "keyword_class_desc", "cluster_label", "cluster_label_desc"],\n            errors="ignore", inplace=True)\n\n    return df\n')


# In[ ]:


from google.colab import files  # only for Colab
uploaded = files.upload()        # select updated_logs.csv from your computer
import pandas as pd

# Lire le fichier fusionné
new_df = pd.read_csv("updated_logs.csv")

# Vérification
print("✅ new_df chargé avec", new_df.shape[0], "lignes et", new_df.shape[1], "colonnes")
new_df.head()
new_df["datetime"] = pd.to_datetime(new_df["datetime"], errors="coerce")

# Mettre datetime comme index
new_df = new_df.set_index("datetime")


# In[ ]:


import os
import math
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier, XGBRegressor
import joblib

# ------------------------
# 1) Label engineering
# ------------------------
def make_incident_targets(df, incident_condition=None, minutes_windows=[1, 5]):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have DatetimeIndex")

    if incident_condition is None:
        def incident_condition(row):
            return (str(row.get('level', '')).lower() not in ['info', 'notice']) and isinstance(row.get('final_label', None), str)

    is_incident = df.apply(incident_condition, axis=1).astype(bool)
    incident_times = df.index[is_incident].values
    idx_ts = df.index.values

    next_incident_times = []
    for t in idx_ts:
        pos = np.searchsorted(incident_times, t, side='right')
        if pos < len(incident_times):
            next_incident_times.append(pd.Timestamp(incident_times[pos]))
        else:
            next_incident_times.append(pd.NaT)

    df = df.copy()
    df['next_incident_time'] = pd.to_datetime(next_incident_times)
    df['time_to_next_incident_seconds'] = (df['next_incident_time'] - df.index.to_series()).dt.total_seconds()
    df['time_to_next_incident_minutes'] = df['time_to_next_incident_seconds'] / 60.0

    for m in minutes_windows:
        col = f'incident_within_{m}min'
        df[col] = ((df['time_to_next_incident_seconds'] <= m * 60) & (~df['time_to_next_incident_seconds'].isna())).astype(int)

    return df

# ------------------------
# 2) Helper
# ------------------------
def flatten_1d(x):
    return np.array(x).ravel()

# ✅ Picklable keyword transformer
class KeywordMatcher(BaseEstimator, TransformerMixin):
    def __init__(self, keywords):
        self.keywords = keywords
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        s = pd.Series(np.array(X).ravel())
        arr = np.column_stack([s.str.contains(k, case=False, na=False).astype(int) for k in self.keywords])
        return arr

# ------------------------
# 3) Preprocessing pipeline
# ------------------------
def build_preprocessor(num_features, cat_features, text_feature, keywords, tfidf_max_features=500):
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scale', StandardScaler())
    ])

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('ohe', ohe)
    ])

    text_pipe = Pipeline([
        ('flatten', FunctionTransformer(flatten_1d, validate=False)),
        ('tfidf', TfidfVectorizer(max_features=tfidf_max_features))
    ])

    kw_transformer = KeywordMatcher(keywords)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_features),
            ('cat', cat_pipe, cat_features),
            ('text', text_pipe, text_feature),
            ('kw', kw_transformer, text_feature)
        ],
        remainder='drop',
        sparse_threshold=0.0
    )
    return preprocessor

# ------------------------
# 4) Incident Type Classifier
# ------------------------
def train_incident_type_classifier(df,
                                   num_features,
                                   cat_features,
                                   text_feature,
                                   keywords,
                                   tfidf_max_features,
                                   save_prefix="models/alert"):
    mask = df["final_label"].notna() & (~df["level"].str.lower().isin(["info", "notice"]))
    df_type = df.loc[mask].copy()

    if df_type.empty:
        print("⚠️ No incident type labels found, skipping incident-type training.")
        return None, None, None

    lbl = LabelEncoder()
    y_type = lbl.fit_transform(df_type["final_label"])

    feat_cols = list(num_features) + list(cat_features) + [text_feature]
    X_type_df = df_type[feat_cols].copy()

    preproc_type = build_preprocessor(num_features, cat_features, text_feature, keywords, tfidf_max_features)
    X_type = preproc_type.fit_transform(X_type_df)

    clf_type = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="mlogloss",
        n_jobs=4
    )
    clf_type.fit(X_type, y_type)

    print(f"✅ Incident type classifier trained on {len(df_type)} samples. Classes: {list(lbl.classes_)}")
    return clf_type, lbl, preproc_type

# ------------------------
# 5) Main Training function
# ------------------------
def train_early_alert_and_ttr(new_df,
                              minutes_window=5,
                              tfidf_max_features=500,
                              keywords=None,
                              num_features=None,
                              cat_features=None,
                              text_feature='cleaned_message',
                              n_splits=5,
                              xgb_clf_params=None,
                              xgb_reg_params=None,
                              quantile_alphas=None,
                              cap_minutes=180,
                              save_prefix='models/alert'):
    save_dir = os.path.dirname(save_prefix)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    df = new_df.sort_index()

    if num_features is None:
        num_features = [
            'epoch','time_diff_prev','time_diff_server','logs_15min_count',
            'warn_15min_count','msg_len_mean_10','msg_len_std_10',
            'message_length','hour'
        ]
    if cat_features is None:
        cat_features = ['level','component','machine','server','thread','kernel_info','code']
    if keywords is None:
        keywords = ['exception','timeout','failed','refused','crash','stacktrace']

    for c in num_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Label engineering
    df = make_incident_targets(df, incident_condition=None, minutes_windows=[1, 5, 10])
    target_col = f'incident_within_{minutes_window}min'

    if text_feature not in df.columns:
        df[text_feature] = ''

    feat_cols = list(num_features) + list(cat_features) + [text_feature]
    X_df = df[feat_cols].copy()
    y_clf = df[target_col].astype(int).values

    # Regression target
    reg_mask = ~df['time_to_next_incident_minutes'].isna()
    X_reg_df = df.loc[reg_mask, feat_cols].copy()
    y_reg_minutes = df.loc[reg_mask, 'time_to_next_incident_minutes'].values
    y_reg_minutes = np.minimum(y_reg_minutes, cap_minutes)
    y_reg = np.log1p(y_reg_minutes)

    preproc = build_preprocessor(num_features, cat_features, text_feature, keywords, tfidf_max_features)
    X_all = preproc.fit_transform(X_df)
    X_reg = preproc.transform(X_reg_df) if X_reg_df.shape[0] > 0 else np.empty((0, X_all.shape[1]))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if xgb_clf_params is None:
        xgb_clf_params = dict(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=4
        )
    if xgb_reg_params is None:
        xgb_reg_params = dict(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            objective='reg:squarederror',
            verbosity=0,
            n_jobs=4
        )

    clf = XGBClassifier(**xgb_clf_params)

    # --- Classification training
    try:
        splits = list(tscv.split(X_all))
    except Exception:
        splits = []

    clf.fit(X_all, y_clf, verbose=False)

    # --- Regression training
    quantile_models = {}
    if X_reg.shape[0] > 0:
        if quantile_alphas is None:
            reg = XGBRegressor(**xgb_reg_params)
            reg.fit(X_reg, y_reg, verbose=False)
        else:
            for alpha in quantile_alphas:
                q_params = xgb_reg_params.copy()
                q_params.update(dict(objective="reg:quantileerror", alpha=alpha))
                qreg = XGBRegressor(**q_params)
                qreg.fit(X_reg, y_reg, verbose=False)
                quantile_models[alpha] = qreg
            reg = None
    else:
        reg = None

    # Incident type classifier
    clf_type, lbl, preproc_type = train_incident_type_classifier(
        df=df,
        num_features=num_features,
        cat_features=cat_features,
        text_feature=text_feature,
        keywords=keywords,
        tfidf_max_features=tfidf_max_features,
        save_prefix=save_prefix
    )

    # ------------------------
    # Save (use consistent filenames)
    # ------------------------
    save_prefix = save_prefix.rstrip('/')  # ensure no trailing slash

    # preprocessor & classifier
    joblib.dump(preproc, f"{save_prefix}_preprocessor.joblib")
    joblib.dump(clf, f"{save_prefix}_clf.joblib")

    # regressor or quantiles
    if reg is not None:
        joblib.dump(reg, f"{save_prefix}_reg.joblib")
    if quantile_models:
        for alpha, qreg in quantile_models.items():
            joblib.dump(qreg, f"{save_prefix}_reg_q{alpha}.joblib")

    # Incident-type artifacts (if trained)
    if clf_type is not None:
        joblib.dump(preproc_type, f"{save_prefix}_type_preprocessor.joblib")
        joblib.dump(clf_type, f"{save_prefix}_type_clf.joblib")
    if lbl is not None:
        joblib.dump(lbl, f"{save_prefix}_label_encoder.joblib")

    print("✅ Saved artifacts with prefix:", save_prefix)

    # --- Summary
    print("\n================ Model Performance Summary ================")
    print("✅ Early alert classifier, regressor, and incident type models saved in models/")

    return {
        'preprocessor': preproc,
        'classifier': clf,
        'regressor': reg,
        'quantile_models': quantile_models if quantile_models else None,
        'incident_type_clf': clf_type,
        'label_encoder': lbl
    }

# ------------------------
# Example usage
# ------------------------
results = train_early_alert_and_ttr(new_df, minutes_window=5 ,save_prefix="models/alert")


# In[ ]:


import os
print(os.listdir("models"))


# In[ ]:


# Install required packages (may take a minute)
get_ipython().system('pip install -q streamlit pyngrok joblib scikit-learn xgboost pandas numpy')
# For neatness: upgrade pyngrok if needed
get_ipython().system('pip install -q --upgrade pyngrok')


import os
os.environ["NGROK_AUTHTOKEN"] = "NGROK_KEY"


# In[ ]:


from google.colab import files

uploaded = files.upload()


# In[ ]:


get_ipython().system('pip install streamlit-autorefresh')


# In[ ]:


# Fermer tous les processus ngrok ouverts
get_ipython().system('pkill -f ngrok')


# In[ ]:


get_ipython().system('pip install scikit-learn==1.6.1')


# In[ ]:


from pyngrok import ngrok
public_url = ngrok.connect(8501)
print("🌍 Streamlit app URL:", public_url)

get_ipython().system('streamlit run app.py --server.port 8501 --server.address 0.0.0.0')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', '#for description part\nimport streamlit as st\nimport concurrent.futures\nimport threading\nimport copy\nimport pandas as pd\nimport numpy as np\nimport joblib\nimport time\nimport os\nimport re\nimport spacy\nimport nltk\nimport requests\nimport json\nimport random\nfrom queue import Queue\nimport tempfile\nfrom nltk.corpus import stopwords\nfrom sentence_transformers import SentenceTransformer\nfrom sklearn.preprocessing import StandardScaler\nfrom datetime import datetime\nfrom collections import Counter\nfrom functools import lru_cache\n\n# ===========================\n# Import from your pipeline\n# ===========================\ntry:\n    from my_pipeline import (\n        preprocess_log_line,\n        enrich_log_row,\n        classify_log_row,\n        extract_contextual_insight,\n        get_sequence_description,\n        build_enriched_input,\n        summarize_with_llama3\n    )\nexcept ImportError:\n    st.error("Could not import functions from my_pipeline. Please make sure my_pipeline.py is in the same directory.")\n\n    # Define fallback functions\n    def preprocess_log_line(line):\n        return {"message": line, "cleaned_message": line[:100]}\n\n    def enrich_log_row(row):\n        return row\n\n    def classify_log_row(row):\n        row["final_label"] = "Unknown"\n        return row\n\n    def extract_contextual_insight(message):\n        return "Sample contextual insight"\n\n    def get_sequence_description(row, full_df):\n        return "Sequence description"\n\n    def build_enriched_input(row):\n        return f"Enriched input: {row.get(\'message\', \'\')}"\n\n    def summarize_with_llama3(text):\n        return f"AI summary for: {text[:100]}..."\n\n# ===========================\n# Incident Report Formatter\n# ===========================\ndef format_professional_summary(row: dict, ai_summary: str, rule_note: str, seq_desc: str) -> str:\n    return (\n        f"\\n🧾 === INCIDENT REPORT ===\\n"\n        f"📅 Date       : {row.get(\'datetime\', \'N/A\')}\\n"\n        f"🖥️ Location   : {row.get(\'machine\', \'N/A\')} | Server: {row.get(\'server\', \'N/A\')}\\n"\n        f"🧩 Component  : {row.get(\'component\', \'N/A\')} ({row.get(\'component_category\', \'N/A\')})\\n"\n        f"⚠️ Level      : {row.get(\'level\', \'N/A\')} ({row.get(\'level_category\', \'N/A\')})\\n"\n        f"🌟 Classified as : {row.get(\'final_label\', \'N/A\')}\\n"\n        f"\\n🤖 AI-Generated Summary:\\n{ai_summary.strip().capitalize() if ai_summary else \'N/A\'}\\n"\n        f"==========================\\n"\n    )\n\n# Safe summarization function with retries\ndef safe_summarize(input_text: str, retries: int = 3) -> str:\n    for attempt in range(retries):\n        try:\n            return summarize_with_llama3(input_text[:3000])\n        except requests.exceptions.HTTPError as e:\n            if "429" in str(e):\n                wait_time = (2 ** attempt) + random.random()\n                print(f"⏳ Rate limit hit, retrying in {wait_time:.1f}s...")\n                time.sleep(wait_time)\n                continue\n            raise\n        except Exception as e:\n            return f"⚠️ Error generating summary: {e}"\n    return "⚠️ Error generating summary after retries."\n\n# ===========================\n# Processing Functions (Streamlit-compatible)\n# ===========================\ndef process_single_line(line: str, line_num: int, total_lines: int):\n    """Process a single log line and return the report"""\n    try:\n        row_base = preprocess_log_line(line)\n        if row_base is None:\n            return None, line_num\n\n        # Enrich and classify\n        enriched = enrich_log_row(copy.deepcopy(row_base))\n        classified = classify_log_row(copy.deepcopy(row_base))\n        merged = {**row_base, **enriched, **classified}\n\n        # Generate insights and summary\n        rule_note = extract_contextual_insight(merged.get("message", ""))\n        sequence_desc = "Sequence analysis"  # Simplified for streaming\n        input_text = build_enriched_input(merged)\n        ai_summary = safe_summarize(input_text)\n\n        # Format the report\n        report = format_professional_summary(merged, ai_summary, rule_note, sequence_desc)\n        return report, line_num\n\n    except Exception as e:\n        error_report = f"⚠️ Error processing line {line_num}: {str(e)}"\n        return error_report, line_num\n\ndef process_log_file_streamlit(uploaded_file, max_workers_lines: int = 4):\n    """Process log file in a Streamlit-compatible way"""\n    # Read the file content\n    content = uploaded_file.getvalue().decode("utf-8")\n    lines = [line.strip() for line in content.splitlines() if line.strip()]\n    total_lines = len(lines)\n\n    # Initialize progress\n    st.session_state.total_lines = total_lines\n    st.session_state.processed_lines = 0\n    st.session_state.reports = []\n    st.session_state.current_status = "Processing..."\n\n    # Create a progress bar and status area\n    progress_bar = st.progress(0)\n    status_text = st.empty()\n    reports_container = st.container()\n\n    # Process lines with ThreadPoolExecutor\n    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_lines) as executor:\n        # Submit all tasks\n        future_to_line = {\n            executor.submit(process_single_line, line, i, total_lines): i\n            for i, line in enumerate(lines)\n        }\n\n        # Process completed tasks as they come in\n        for future in concurrent.futures.as_completed(future_to_line):\n            line_num = future_to_line[future]\n            try:\n                report, line_num = future.result()\n                if report:\n                    st.session_state.reports.append(report)\n\n                # Update progress\n                st.session_state.processed_lines += 1\n                progress = st.session_state.processed_lines / total_lines\n                progress_bar.progress(progress)\n\n                # Update status\n                status_text.text(f"Processed {st.session_state.processed_lines}/{total_lines} lines")\n\n                # Update reports display in real-time (last 10 reports)\n                with reports_container:\n                    st.markdown("### 📊 Real-time Incident Reports")\n\n                    # Custom CSS for better styling\n                    st.markdown("""\n                    <style>\n                    .report-container {\n                        background: linear-gradient(135deg, #2c3e50, #34495e);\n                        border: 2px solid #3498db;\n                        border-radius: 10px;\n                        padding: 15px;\n                        margin: 10px 0;\n                        color: #ecf0f1;\n                        font-family: \'Arial\', sans-serif;\n                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);\n                    }\n                    .report-container h3 {\n                        color: #3498db;\n                        border-bottom: 2px solid #3498db;\n                        padding-bottom: 5px;\n                        margin-top: 0;\n                    }\n                    .report-container h4 {\n                        color: #2ecc71;\n                        margin-bottom: 5px;\n                    }\n                    .report-title {\n                        color: #e74c3c;\n                        font-weight: bold;\n                        font-size: 1.2em;\n                    }\n                    .report-field {\n                        color: #bdc3c7;\n                        margin: 3px 0;\n                    }\n                    .report-value {\n                        color: #ecf0f1;\n                        font-weight: bold;\n                    }\n                    </style>\n                    """, unsafe_allow_html=True)\n\n                    for report in st.session_state.reports[-10:]:  # Show last 10 reports\n                        # Enhanced HTML formatting with better styling\n                        html_report = report.replace("\\n", "<br>")\n\n                        # Replace sections with styled divs\n                        html_report = html_report.replace(\n                            "🧾 === INCIDENT REPORT ===",\n                            \'<div class="report-title">🧾 INCIDENT REPORT</div>\'\n                        )\n                        html_report = html_report.replace(\n                            "🤖 AI-Generated Summary:",\n                            \'<div class="report-title">🤖 AI-Generated Summary:</div>\'\n                        )\n                        html_report = html_report.replace(\n                            "📅 Date       : ",\n                            \'<span class="report-field">📅 Date:</span> <span class="report-value">\'\n                        )\n                        html_report = html_report.replace(\n                            "🖥️ Location   : ",\n                            \'<br><span class="report-field">🖥️ Location:</span> <span class="report-value">\'\n                        )\n                        html_report = html_report.replace(\n                            "🧩 Component  : ",\n                            \'<br><span class="report-field">🧩 Component:</span> <span class="report-value">\'\n                        )\n                        html_report = html_report.replace(\n                            "⚠️ Level      : ",\n                            \'<br><span class="report-field">⚠️ Level:</span> <span class="report-value">\'\n                        )\n                        html_report = html_report.replace(\n                            "🌟 Classified as : ",\n                            \'<br><span class="report-field">🌟 Classified as:</span> <span class="report-value">\'\n                        )\n\n                        # Close the value spans and add proper formatting\n                        html_report = html_report.replace("N/A", "N/A</span>")\n                        html_report = html_report.replace(")<br>", ")</span><br>")\n                        html_report = html_report.replace("<br><br>", "</span><br><br>")\n\n                        # Final cleanup\n                        html_report = html_report.replace("==========================", "<hr style=\'border-color: #3498db;\'>")\n\n                        st.markdown(\n                            f"""<div class="report-container">{html_report}</div>""",\n                            unsafe_allow_html=True\n                        )\n\n            except Exception as e:\n                error_msg = f"⚠️ Error processing line {line_num}: {str(e)}"\n                st.session_state.reports.append(error_msg)\n\n    # Mark as complete\n    st.session_state.current_status = "Complete!"\n    status_text.success("✅ Processing complete!")\n    progress_bar.progress(1.0)\n\n# ===========================\n# Streamlit UI\n# ===========================\ndef main():\n    st.set_page_config(page_title="Log Incident Reporting System", layout="wide")\n\n    # Add custom CSS for overall app styling\n    st.markdown("""\n    <style>\n    .main {\n        background-color: #1e1e1e;\n        color: #ffffff;\n    }\n    .stApp {\n        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);\n    }\n    .css-1d391kg, .css-1y4v7go {\n        background-color: #2c3e50 !important;\n    }\n    .report-final-container {\n        background: linear-gradient(135deg, #2c3e50, #34495e);\n        border: 2px solid #3498db;\n        border-radius: 10px;\n        padding: 20px;\n        margin: 15px 0;\n        color: #ecf0f1;\n        font-family: \'Arial\', sans-serif;\n        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.4);\n    }\n    .report-final-title {\n        color: #3498db;\n        font-size: 1.3em;\n        font-weight: bold;\n        border-bottom: 2px solid #3498db;\n        padding-bottom: 8px;\n        margin-bottom: 15px;\n    }\n    </style>\n    """, unsafe_allow_html=True)\n\n    st.title("📊 Log Incident Reporting System")\n    st.write("Generate professional incident reports from log files")\n\n    # Initialize session state\n    if \'processing\' not in st.session_state:\n        st.session_state.processing = False\n    if \'reports\' not in st.session_state:\n        st.session_state.reports = []\n    if \'processed_lines\' not in st.session_state:\n        st.session_state.processed_lines = 0\n    if \'total_lines\' not in st.session_state:\n        st.session_state.total_lines = 0\n    if \'current_status\' not in st.session_state:\n        st.session_state.current_status = "Ready"\n\n    # File uploader\n    uploaded_file = st.file_uploader("Choose a log file", type=[\'log\', \'txt\'])\n\n    # Processing options\n    st.sidebar.header("Processing Options")\n    max_workers_lines = st.sidebar.slider("Processing threads", 1, 8, 4)\n\n    # Display current status\n    st.sidebar.header("Status")\n    st.sidebar.write(f"**Status:** {st.session_state.current_status}")\n    if st.session_state.total_lines > 0:\n        st.sidebar.write(f"**Progress:** {st.session_state.processed_lines}/{st.session_state.total_lines} lines")\n        st.sidebar.write(f"**Reports generated:** {len(st.session_state.reports)}")\n\n    # Start Processing button\n    if uploaded_file is not None and not st.session_state.processing:\n        if st.button("Start Processing", type="primary"):\n            st.session_state.processing = True\n            st.session_state.reports = []\n\n            # Process the file\n            process_log_file_streamlit(uploaded_file, max_workers_lines)\n            st.session_state.processing = False\n\n    # Stop Processing button\n    if st.session_state.processing:\n        if st.button("Stop Processing"):\n            st.session_state.processing = False\n            st.session_state.current_status = "Stopped by user"\n            st.rerun()\n\n    # Display final results if processing is complete\n    if not st.session_state.processing and st.session_state.reports:\n        st.success(f"✅ Processing complete! Generated {len(st.session_state.reports)} incident reports.")\n\n        # Display all reports in an expandable section with better styling\n        with st.expander("View All Incident Reports", expanded=True):\n            for i, report in enumerate(st.session_state.reports):\n                # Enhanced HTML formatting for final display\n                html_report = report.replace("\\n", "<br>")\n\n                # Apply the same styling as real-time reports but with final container class\n                html_report = html_report.replace(\n                    "🧾 === INCIDENT REPORT ===",\n                    \'<div class="report-final-title">🧾 INCIDENT REPORT</div>\'\n                )\n                html_report = html_report.replace(\n                    "🤖 AI-Generated Summary:",\n                    \'<div style="color: #2ecc71; font-weight: bold; margin: 10px 0;">🤖 AI-Generated Summary:</div>\'\n                )\n                html_report = html_report.replace(\n                    "📅 Date       : ",\n                    \'<div style="color: #bdc3c7; margin: 3px 0;">📅 Date: <span style="color: #ecf0f1; font-weight: bold;">\'\n                )\n                html_report = html_report.replace(\n                    "🖥️ Location   : ",\n                    \'<br>🖥️ Location: <span style="color: #ecf0f1; font-weight: bold;">\'\n                )\n                html_report = html_report.replace(\n                    "🧩 Component  : ",\n                    \'<br>🧩 Component: <span style="color: #ecf0f1; font-weight: bold;">\'\n                )\n                html_report = html_report.replace(\n                    "⚠️ Level      : ",\n                    \'<br>⚠️ Level: <span style="color: #ecf0f1; font-weight: bold;">\'\n                )\n                html_report = html_report.replace(\n                    "🌟 Classified as : ",\n                    \'<br>🌟 Classified as: <span style="color: #ecf0f1; font-weight: bold;">\'\n                )\n\n                # Close the value spans\n                html_report = html_report.replace("N/A", "N/A</span>")\n                html_report = html_report.replace(")<br>", ")</span><br>")\n                html_report = html_report.replace("<br><br>", "</span><br><br>")\n                html_report = html_report.replace("==========================", "<hr style=\'border-color: #3498db; margin: 15px 0;\'>")\n\n                st.markdown(\n                    f"""<div class="report-final-container">{html_report}</div>""",\n                    unsafe_allow_html=True\n                )\n\n    # Display ready message when no file is uploaded\n    if uploaded_file is None:\n        st.info("📁 Please upload a log file to begin processing")\n\nif __name__ == "__main__":\n    main()\n')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', '#for predection\nimport tempfile\nimport streamlit as st\nimport threading\nimport time\nimport queue\nimport os\nimport joblib\nimport pandas as pd\nimport numpy as np\nfrom sklearn.base import BaseEstimator, TransformerMixin\nfrom streamlit_autorefresh import st_autorefresh\n\nfrom my_pipeline import (\n    preprocess_log_line,\n    enrich_log_row,\n    classify_log_row,\n)\n\n# ==========================================\n# 🧩 Keyword Matcher (if used in preprocessing)\n# ==========================================\nclass KeywordMatcher(BaseEstimator, TransformerMixin):\n    def __init__(self, keywords=None):\n        self.keywords = keywords or []\n\n    def fit(self, X, y=None):\n        return self\n\n    def transform(self, X):\n        s = pd.Series(np.array(X).ravel())\n        arr = np.column_stack([\n            s.str.contains(k, case=False, na=False).astype(int)\n            for k in self.keywords\n        ])\n        return arr\n\n\ndef flatten_1d(X):\n    return X.to_numpy() if hasattr(X, "to_numpy") else X\n\n\n# ==========================================\n# 📦 Load Models\n# ==========================================\ndef load_model(path):\n    try:\n        return joblib.load(path)\n    except Exception as e:\n        st.error(f"❌ Error loading {path}: {e}")\n        return None\n\n\nmodels = {\n    "preproc": load_model("models/alert_preprocessor.joblib"),\n    "clf": load_model("models/alert_clf.joblib"),\n    "reg": load_model("models/alert_reg.joblib"),\n    "type_preproc": load_model("models/alert_type_preprocessor.joblib"),\n    "type_clf": load_model("models/alert_type_clf.joblib"),\n    "label_encoder": load_model("models/alert_label_encoder.joblib"),\n}\n\n\n# ==========================================\n# 🔮 Prediction\n# ==========================================\ndef process_and_predict(df):\n    if df.empty:\n        return 0, False\n\n    if models["preproc"] is None or models["clf"] is None:\n        st.warning("⚠️ Models not loaded, skipping prediction.")\n        return 0, False\n\n    try:\n        feat_cols = [\n            "epoch", "time_diff_prev", "time_diff_server",\n            "logs_15min_count", "warn_15min_count",\n            "msg_len_mean_10", "msg_len_std_10", "message_length",\n            "hour", "level", "component", "machine", "server",\n            "thread", "kernel_info", "code", "cleaned_message"\n        ]\n\n        X = df[feat_cols]\n        X_alert = models["preproc"].transform(X)\n\n        probs = models["clf"].predict_proba(X_alert)[:, 1]\n        alerts = probs > 0.5\n\n        # Regression (ETA)\n        times = None\n        if models["reg"]:\n            times = models["reg"].predict(X_alert)\n\n        # Alert type classification\n        type_pred = None\n        if (\n            models["type_preproc"]\n            and models["type_clf"]\n            and models["label_encoder"]\n        ):\n            X_type = models["type_preproc"].transform(X)\n            type_idx = models["type_clf"].predict(X_type)\n            type_pred = models["label_encoder"].inverse_transform(type_idx)[-1]\n\n        return probs[-1], bool(alerts[-1]), times, type_pred\n\n    except Exception as e:\n        st.error(f"❌ Prediction error: {e}")\n        return 0, False, None, None\n\n\n# ==========================================\n# 🔄 Real-Time Log Processing\n# ==========================================\ndef simulate_realtime_processing(file_path, message_queue, stop_event):\n    buffer = []\n\n    try:\n        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:\n            for line in f:\n                if stop_event.is_set():\n                    break\n\n                parsed = preprocess_log_line(line)\n                if not parsed:\n                    continue\n\n                enriched = enrich_log_row(parsed)\n                classified = classify_log_row(enriched)\n                buffer.append(classified)\n\n                if len(buffer) >= 5:  # batch\n                    df = pd.DataFrame(buffer)\n\n                    prob, is_alert, times, type_pred = process_and_predict(df)\n\n                    if is_alert:\n                        msg = f"🚨 ALERT: p={prob:.2f}"\n                        if type_pred:\n                            msg += f" | Type={type_pred}"\n                        if times is not None:\n                            msg += f" | ETA={times[-1]:.1f}s"\n                        message_queue.put(("alert", msg))\n                    else:\n                        message_queue.put(("info", f"✅ OK: p={prob:.2f}"))\n\n                    buffer.clear()\n\n                time.sleep(0.5)\n\n    except Exception as e:\n        message_queue.put(("error", f"❌ Error reading log: {e}"))\n\n\n# ==========================================\n# 🖥️ Streamlit UI\n# ==========================================\ndef main():\n    st.set_page_config(page_title="Log Monitoring with ML", layout="wide")\n    st.title("📊 Log Monitoring with ML Alerts")\n\n    st.sidebar.header("⚙️ Controls")\n    uploaded_file = st.sidebar.file_uploader("Upload a .log file", type=["log"])\n\n    if uploaded_file is None:\n        st.info("📂 Upload a log file to start monitoring.")\n        return\n\n    # Save uploaded file locally\n    log_path = "uploaded_log.log"\n    with open(log_path, "wb") as f:\n        f.write(uploaded_file.getbuffer())\n\n    # Global state\n    if "stop_event" not in st.session_state:\n        st.session_state.stop_event = threading.Event()\n\n    if "message_queue" not in st.session_state:\n        st.session_state.message_queue = queue.Queue()\n\n    if ("thread" not in st.session_state) or (not st.session_state.thread.is_alive()):\n        st.session_state.stop_event.clear()\n        st.session_state.thread = threading.Thread(\n            target=simulate_realtime_processing,\n            args=(log_path, st.session_state.message_queue, st.session_state.stop_event),\n            daemon=True\n        )\n        st.session_state.thread.start()\n\n    # Display area\n    st.subheader("📋 Recent Logs / Alerts")\n    log_box = st.empty()\n    messages = []\n\n    while not st.session_state.message_queue.empty():\n        mtype, msg = st.session_state.message_queue.get()\n\n        if mtype == "alert":\n            messages.append(f"🔴 {msg}")\n        elif mtype == "info":\n            messages.append(f"🟢 {msg}")\n        elif mtype == "error":\n            messages.append(f"⚠️ {msg}")\n        else:\n            messages.append(msg)\n\n    if messages:\n        log_box.text("\\n".join(messages[-15:]))\n\n    # Auto refresh every 2 seconds\n    st_autorefresh(interval=2000, limit=None)\n\n\n# ==========================================\n# 🚀 Launch App\n# ==========================================\nif __name__ == "__main__":\n    main()\n')

