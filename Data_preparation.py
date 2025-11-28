#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import files

uploaded = files.upload()


# In[ ]:


import re
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Download required resources
nltk.download('stopwords', quiet=True)

# Load NLP tools
nlp = spacy.load("en_core_web_sm")
nltk_stopwords = set(stopwords.words('english'))
spacy_stopwords = nlp.Defaults.stop_words
combined_stopwords = spacy_stopwords.union(nltk_stopwords)

# Load sentence transformer model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Regex to parse logs
log_pattern = re.compile(
    r"####<(?P<timestamp>[^>]+)>\s*"
    r"<(?P<level>[^>]+)>\s*"
    r"<(?P<component>[^>]+)>\s*"
    r"<(?P<machine>[^>]+)>\s*"
    r"<(?P<server>[^>]+)>\s*"
    r"<(?P<thread>.*?)>\s*"
    r"<(?P<kernel_info>.*?)>\s*"
    r"<>\s*<>\s*"  # Skipped empty fields
    r"<(?P<epoch>\d+)>\s*"
    r"<(?P<code>bea-\d+)>\s*"
    r"<(?P<message>.*?)(?:>)?\s*$"
)

def clean_log_message(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z ]+', ' ', text.lower())
    doc = nlp(text)
    return ' '.join([
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_ not in combined_stopwords and len(token.lemma_) > 1
    ])

def preprocess_logs(raw_lines, n_clusters=20):
    # Clean lines
    cleaned = [line.strip().lower() for line in raw_lines if line.startswith("####<")]

    # Parse structured logs
    parsed_logs = [log_pattern.match(line).groupdict() for line in cleaned if log_pattern.match(line)]
    df = pd.DataFrame(parsed_logs)

    # Text cleaning
    df['cleaned_message'] = df['message'].apply(clean_log_message)

    X_bert = bert_model.encode(df['cleaned_message'].tolist())

    # --- 2. One-hot encode categorical columns
    level_onehot = pd.get_dummies(df['level'], prefix="level")
    component_onehot = pd.get_dummies(df['component'], prefix="component")

    # --- 3. Numeric feature: message length
    df['msg_length'] = df['message'].apply(len)
    msg_length_scaled = StandardScaler().fit_transform(df[['msg_length']])

    # --- 4. Combine features: [BERT | Level | Component | Length]
    X_combined = np.hstack((
        X_bert,
        level_onehot.values,
        component_onehot.values,
        msg_length_scaled
    ))

    # --- 5. Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_combined)

    return df, X_bert


# In[ ]:


import pandas as pd
import numpy as np
import re

def enrich_log_dataframe(df):
    # ----------------------------
    # 1. Log level mapping
    # ----------------------------
    log_level_categories = {
        'emergency': 'System Down',
        'alert': 'Security Breach Detected',
        'critical': 'Service Outage',
        'error': 'Operation Failed',
        'warning': 'Potential Issue',
        'warnning': 'Potential Issue',   # typo case
        'notice': 'Significant Event',
        'info': 'Normal Operation',
        'debug': 'Troubleshooting',
        'trace': 'Execution Flow',
        'failure': 'Critical System Failure',
        'fatal': 'Process Termination',
        'j2ee_error': 'Container Failure',
        'deployment_failure': 'Deployment Aborted',
        'tx_rollback': 'Transaction Failed',
        'unavailable': 'Admin Server Down',
        'restarted': 'Server Rebooted',
        'recoverable': 'Recoverable Error Occurred',
        'suspicious': 'Suspicious Activity Detected',
        'emergency_shutdown': 'Emergency System Shutdown',
        'startup': 'System Initialization',
        'shutdown': 'System Shutdown',
        'maintenance': 'Scheduled System Maintenance',
    }

    # ----------------------------
    # 2. Component regex mapping
    # ----------------------------
    component_categories = {
        r'\bj2ee\b': 'j2ee container',
        r'\bejbs?\b': 'enterprise java beans',
        r'\bservlet\b': 'web component',
        r'\bjsp\b': 'presentation layer',
        r'\bjndi\b': 'naming services',
        r'\bjca\b': 'connector architecture',
        r'weblogicserver': 'core server',
        r'security': 'authentication & authorization',
        r'deployer': 'deployment engine',
        r'jdbc': 'database connectivity',
        r'\bjms\b': 'messaging services',
        r'\bcluster\b': 'server clustering',
        r'console': 'management interface',
        r'nodemanager': 'server instance controller',
        r'workmanager': 'thread pool management',
        r'harvester': 'metric collection',
        r'stdout': 'standard output logging',
        r'appmerge': 'application packaging',
        r'diagnostics': 'monitoring subsystem',
        r'coherence': 'in-memory data grid',
        r'webapp': 'web application framework',
        r'wlst': 'scripting tool',
        r'snmp': 'network management',
        r'connector': 'adapter integration',
        r'jvm|jvm crash': 'Java Virtual Machine',
        r'datasource': 'Database Connection Pool',
        r'ssl|tls|keystore|truststore': 'Secure Socket Layer',
        r'webcontainer|servlet engine': 'Web Container',
        r'\bdb\b|sql|query|cursor': 'Database Service',
        r'weblogic server|server crash': 'WebLogic Server',
        r'clustering|splitbrain|partition': 'Clustered Environment',
        r'loadbalancer|balancer': 'Load Balancer',
        r'cache|coherence': 'Memory Caching',
        r'gc|garbage collection': 'JVM Garbage Collection',
        r'socket|port|connection refused|handshake': 'Network Socket',
        r'storage|store|filesystem|disk|i/o|nfs|volume|partition|quota': 'Storage System',
        r'memory|heap|outofmemory|oom': 'Memory Issues',
        r'overflow|buffer|exhausted': 'Buffer Overflow',
        r'connection|timeout|refused|reset': 'Connection Issues',
        r'crash|abort|failure|exception|error': 'Critical Failures',
        r'corruption|corrupted': 'Data Corruption',
        r'invalid|illegal|unsupported|unauthorized|violation': 'Invalid Operation',
        r'breach|intrusion|attack|malicious': 'Security Breaches',
        r'lock|constraint|deadlock': 'Locking Issues',
        r'disconnect|unreachable|reset': 'Network Disconnects',
        r'heartbeat|heartbeat failure|timeout': 'Heartbeat Issues',
        r'caching|cache|outofmemory|threshold': 'Cache Issues',
        r'redeployment|undeployment|deploy': 'Deployment Issues',
        r'rollback|commit|transaction': 'Transaction Issues',
        r'queue|consumer|producer|broker|topic': 'Messaging Issues',
        r'io|filesystem|disk|nfs|storage': 'File System Issues',
        r'authentication|authorization|session': 'Authentication Issues',
    }

    # ----------------------------
    # 3. Datetime parsing
    # ----------------------------
    df['datetime'] = pd.to_datetime(df['epoch'].astype(str), errors='coerce', unit='ms')
    df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()

    # ----------------------------
    # 4. Categories
    # ----------------------------
    df['level_category'] = df['level'].str.lower().map(
        {k.lower(): v for k, v in log_level_categories.items()}
    ).fillna('Other')

    df['component_category'] = df['component'].apply(
        lambda x: next(
            (v for k, v in component_categories.items()
             if re.search(k, str(x), re.IGNORECASE)),
            'Other Component'
        )
    )

    # ----------------------------
    # 5. Derived features
    # ----------------------------
    df['message_length'] = df['message'].astype(str).str.len().fillna(0)
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['time_interval'] = df['datetime'].dt.floor('15min')

    # --- Time delta features ---
    df['time_diff_prev'] = df['datetime'].diff().dt.total_seconds().fillna(0)
    df['time_diff_server'] = df.groupby('server')['datetime'].diff().dt.total_seconds().fillna(0)

    # --- Frequency features (rolling counts in 15min windows) ---
    df['logs_15min_count'] = df.groupby('server')['datetime'].transform(
        lambda x: x.rolling('15min').count()
    )
    df['warn_15min_count'] = (df['level'].str.lower().eq('warning')
                              .groupby(df['server'])
                              .transform(lambda x: x.rolling(50, min_periods=1).sum()))

    # --- Statistical features ---
    df['msg_len_mean_10'] = df['message_length'].rolling(10, min_periods=1).mean()
    df['msg_len_std_10'] = df['message_length'].rolling(10, min_periods=1).std().fillna(0)

    # Placeholder for text vectorization (to be added later)
    df['text_feature'] = df['cleaned_message'].astype(str)
    df['peak_hours'] = df['hour'].between(8,18).astype(int)

    return df


# In[ ]:


from collections import Counter
from functools import lru_cache
import pandas as pd
import requests
import os
import re
import json
from dotenv import load_dotenv


load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "API_KEY")
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def call_groq(prompt: str):
    """Call LLaMA 3 on Groq for zero-shot classification."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
    "model": GROQ_MODEL,
    "messages": [
        {"role": "system", "content": "You are a log classifier. Classify the input based on predefined labels."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.2,
    "max_tokens": 512
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# Candidate labels
candidate_labels_by_level = {
    "NOTICE": {
        "connection closed": "Connection was closed, creating a new one",
        "card list request": "Web service call to retrieve the card list",
        "operation card listing": "Operation Card Listing executed with row count",
        "sms webservice url call": "Accessed URL of SMS Express WebService alerts",
        "expiration date logged": "Expiration date recorded or processed",
        "multiple cards returned": "Card listing operation returned >1 rows",
        "server starting": "WebLogic server is starting up",
        "server started in development mode": "Server started in development mode",
        "server started in production mode": "Server started in production mode",
        "server shutdown initiated": "Shutdown of WebLogic server has started",
        "shutdown class executed": "Shutdown class executed successfully",
        "server state changed to running": "Server transitioned to RUNNING state",
        "server state changed to standby": "Server transitioned to STANDBY state",
        "server health changed to ok": "Server health marked as OK",
        "machine reachable": "Machine reachable via Node Manager",
        "node manager started": "Node Manager started and listening",
        "cluster member added": "New member joined WebLogic cluster",
        "cluster heartbeat received": "Heartbeat received from cluster member",
        "jms server activated": "JMS server has been activated successfully",
        "jms server paused": "JMS server paused temporarily",
        "jms destination restarted": "JMS destination was restarted cleanly",
        "jdbc driver registered": "JDBC driver registered without issues",
        "datasource active": "JDBC datasource is active and accepting connections",
        "diagnostic volume set": "Diagnostic volume level set (Low, Medium, High)",
        "log file rotated": "Log rotation triggered and completed",
        "audit log initialized": "Security audit log initialized successfully",
        "security realm initialized": "Security realm configuration loaded",
        "default keystore loaded": "Default identity keystore has been loaded",
        "webapp initialized": "Web application successfully initialized",
        "listener started": "Network listener started and bound to port",
        "application update received": "Application update task started",
        "deployment task completed": "Deployment task completed successfully",
        "module prepared": "Module prepared for deployment",
        "module activated": "Module successfully activated",
        "library referenced": "Shared library successfully referenced",
        "wlst command executed": "WLST command was executed without error",
        "configuration changes saved": "Domain configuration saved via console/WLST",
        "startup class executed": "Startup class has run without error",
        "log broadcast received": "Log broadcast message received by admin server",
    },

    "INFO": {
        "application deployment started": "Application deployment initiated on WebLogic",
        "application deployed successfully": "Application deployed and active",
        "jdbc driver loaded": "JDBC driver loaded and registered",
        "jdbc datasource initialized": "Datasource pool initialized and ready",
        "thread pool statistics": "Periodic thread pool statistics logged",
        "garbage collection summary": "Garbage collection completed; summary logged",
        "ssl certificate loaded": "SSL certificate loaded into keystore",
        "session replication successful": "HTTP session replication completed without errors",
        "jms queue threshold reached": "JMS queue depth reached configured threshold",
        "cluster address resolved": "ClusterAddress DNS resolution successful",
    },

    "WARNING": {
        # Standard WebLogic warnings
        "stuck threads detected": "One or more threads marked STUCK in WebLogic",
        "thread pool at capacity": "Thread pool requests queued; capacity reached",
        "high heap usage": "Heap usage exceeded warning threshold",
        "long gc pause": "GC pause time exceeded tuning target",
        "datasource connection leak": "Potential JDBC connection leak detected",
        "session replication failures": "Some sessions failed to replicate to secondary",
        "jms destination paused": "JMS destination automatically paused due to overflow",
        "low disk space": "File store or server volume low on disk space",
        "ssl certificate expiring soon": "SSL certificate will expire soon; renewal advised",
        "node manager unreachable": "Node Manager not responding to heartbeat",
        "cluster member lost heartbeat": "Cluster member removed due to missed heartbeats",
        "transaction rollback warning": "XA transaction may roll back due to timeout",
        "unauthorized access attempt": "Login failed; invalid credentials or role",
        "dns resolution warnings": "Intermittent DNS resolution failures detected",

        # Custom / additional warnings
        "application exception": "Application has thrown exception, unwinding now",
        "datasource connection released": "Forcibly releasing inactive/harvested connection back into the data source connection pool",

        # Extra (optional but frequent)
        "server state change": "Server state changed to running",
        "jdbc driver registered": "JDBC driver registered",
    },

    "ERROR": {
        "server failed to start": "WebLogic server startup failed; see stack trace",
        "deployment failed": "Application deployment failed on WebLogic server",
        "out of memory errors": "JVM threw OutOfMemoryError",
        "stuck thread time exceeded": "Stuck thread BEA-000337 exceeded MaxTime",
        "jdbc connection failed": "Failed to obtain JDBC connection from datasource",
        "connection pool exhausted": "All connections in pool are in use",
        "transaction timeout": "JTA transaction timed out and rolled back",
        "jms message loss": "Messages lost or undeliverable in JMS server",
        "file store full": "Persistent store cannot write; disk full",
        "ssl handshake failed": "SSL handshake exception; certificate or protocol mismatch",
        "cluster communication failure": "Unable to send/receive multicast or unicast messages",
        "node manager failed": "Node Manager encountered fatal error",
        "authentication failures": "User authentication failed repeatedly",
        "authorization failures": "User lacks required security role or policy",
        "logging failures": "Unable to write to log file; I/O error",
        "port binding errors": "Port already in use; server cannot listen",
        "dns resolution failures": "DNS name could not be resolved",
        "external service unavailability": "Dependent external service is down or unreachable",
        "servlet loading issues": "Servlet failed to initialize correctly",
        "jsp compilation errors": "JSP compilation failed during runtime",
        "ejb lookup failures": "JNDI lookup for EJB failed",
        "resource allocation failures": "Failed to allocate required system resources",
        "health state failed": "Server health transitioned to FAILED",
        "persistent store corruption": "File store corruption detected; recovery needed",
    },

    "DEBUG": {
        "debug security audit": "Verbose security audit logs enabled",
        "debug jdbc driver": "Detailed JDBC driver debugging output",
        "debug jms transport": "Low-level JMS transport debug messages",
        "debug cluster messaging": "Cluster multicast & unicast debug logs",
        "debug memory tracker": "Heap and native memory tracker debug output",
        "debug jndi lookup": "Detailed JNDI lookup tracing",
        "debug webservice invocation": "SOAP / REST client debug messages",
        "debug transaction tracing": "Fine-grained JTA/XA transaction tracing",
    },

    "CRITICAL": {
        "server panic": "WebLogic server issued panic and will exit",
        "jvm crash": "HotSpot JVM crash detected; hs_err dump generated",
        "heap dump generated": "Automatic heap dump due to OOM or panic",
        "persistent store corruption": "Critical corruption in persistent store files",
        "disk full": "Disk full; critical services cannot proceed",
        "database unreachable": "Database unreachable for extended period",
        "cluster master lost": "Cluster master failed; failover unsuccessful",
        "license expired": "WebLogic license expired or invalid",
        "fatal native memory leak": "Native memory leak exceeded safety threshold",
        "fatal thread deadlock": "Fatal deadlock detected among JVM threads",
        "security breach detected": "Potential security breach; immediate action required",
        "unsatisfied link error": "Native library load failure caused fatal error",
        "port binding failure critical": "Essential port binding failed; server exits",
    }
}


@lru_cache(maxsize=None)
def cached_classify(text: str, level: str):
    """Classify logs by restricting to candidate labels of given level first."""

    level = (level or "").strip().upper()
    label_dict = candidate_labels_by_level.get(level, {})

    # Use only level-specific labels first
    candidates = list(label_dict) if label_dict else [
        label for group in candidate_labels_by_level.values() for label in group
    ]

    prompt = f"""You are given a log summary: "{text}"

Choose the most appropriate label from the following list of labels:
{', '.join(candidates)}

Return only the best-matching label from the list (use exact wording from the list)."""

    best_label = call_groq(prompt).strip().lower()

    # --- Matching strategy ---
    # 1. Exact match inside level dict
    for label in label_dict:
        if best_label == label.lower():
            return label, label_dict[label]

    # 2. Case-insensitive substring inside level dict
    for label in label_dict:
        if best_label in label.lower() or label.lower() in best_label:
            return label, label_dict[label]

    # 3. Fallback: check all levels
    for level_group, group_dict in candidate_labels_by_level.items():
        for label in group_dict:
            if best_label == label.lower():
                return label, group_dict[label]
    for level_group, group_dict in candidate_labels_by_level.items():
        for label in group_dict:
            if best_label in label.lower() or label.lower() in best_label:
                return label, group_dict[label]

    # 4. No match -> Ask LLaMA to generate class + description
    message = (text or "").strip()
    cleaned_message = re.sub(r'[^\w\s]', ' ', message)
    cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip().lower()

    llm_prompt = f"""
You are an expert log classifier. A log summary could not be matched to predefined labels.

Original text:
\"\"\"{message}\"\"\"

Cleaned text:
\"\"\"{cleaned_message}\"\"\"

Please propose:
1) A concise label (2–5 words, lowercase, no punctuation) that could act as a new class.
2) A one-sentence human-readable description of what the label means.

Return a JSON object ONLY, exactly like this:
{{"label":"<label>", "description":"<one-sentence description>"}}
"""

    try:
        llm_resp = call_groq(llm_prompt)
    except Exception:
        return best_label, "No matching description found."

    label, description = "", ""
    try:
        parsed = json.loads(llm_resp)
        if isinstance(parsed, dict):
            label = parsed.get("label", "")
            description = parsed.get("description", "")
    except Exception:
        m_label = re.search(r'"label"\s*:\s*"([^"]+)"', llm_resp)
        m_desc = re.search(r'"description"\s*:\s*"([^"]+)"', llm_resp)
        if m_label:
            label = m_label.group(1)
            description = m_desc.group(1) if m_desc else ""
        else:
            parts = [ln.strip() for ln in llm_resp.splitlines() if ln.strip()]
            if parts:
                label = parts[0]
                description = " ".join(parts[1:]).strip()

    if not label:
        return best_label, "No matching description found."
    return label.strip(), description.strip() or "No description generated."


def get_cluster_level(df, cluster_id):
    levels = df.loc[df["cluster"] == cluster_id, "level"]
    return levels.mode().iat[0] if not levels.empty else ""


def top_keywords(df, cluster_id):
    texts = df.loc[df["cluster"] == cluster_id, "cleaned_message"]
    return Counter(" ".join(texts).split()).most_common(5)


def classify_logs(df):
    cluster_label_map = {}
    cluster_desc_map = {}

    # Step 1: classify per cluster
    for cl in df["cluster"].dropna().unique():
        level = get_cluster_level(df, cl)
        keywords = [k for k, _ in top_keywords(df, cl)]
        label, desc = cached_classify(" ".join(keywords), level)
        cluster_label_map[cl] = label
        cluster_desc_map[cl] = desc

    df["cluster_label"] = df["cluster"].map(cluster_label_map)
    df["cluster_label_desc"] = df["cluster"].map(cluster_desc_map)

    # Step 2: row-level classification if cluster-level missing
    def classify_with_context(row):
        level = row.get("level", "").strip().upper()
        context = " | ".join(filter(None, (row.get("message"), row.get("component"))))
        return pd.Series(cached_classify(context, level))

    mask = df["cluster_label"].isna()
    df.loc[mask, ["keyword_class", "keyword_class_desc"]] = df.loc[mask].apply(classify_with_context, axis=1)

    # Step 3: Merge cluster and row-level classification
    def merge_labels(row):
        labels, descs = [], []
        if pd.notna(row["cluster_label"]):
            labels.append(row["cluster_label"])
            descs.append(row["cluster_label_desc"])
        if pd.notna(row.get("keyword_class")) and row["keyword_class"] not in labels:
            labels.append(row["keyword_class"])
            descs.append(row["keyword_class_desc"])
        return pd.Series([" + ".join(labels), " | ".join(descs)])

    df[["final_label", "final_label_desc"]] = df.apply(merge_labels, axis=1)
    df.drop(columns=["keyword_class", "keyword_class_desc", "cluster_label", "cluster_label_desc"],
            errors="ignore", inplace=True)

    return df


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_clusters(X_bert, df):
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X_bert)

    plt.figure(figsize=(20, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='tab10')
    plt.title('Log Message Clusters')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.colorbar()
    plt.show()


# In[ ]:


import numpy as np
# Step 1: Read the uploaded file
filename = next(iter(uploaded))  # Get the name of the uploaded file
with open(filename, 'r') as f:
    raw_lines = f.readlines()

# Step 2: Preprocess logs (clustering + embedding)
df, X_bert = preprocess_logs(raw_lines, n_clusters=20)
df = enrich_log_dataframe(df)
df['datetime'] = pd.to_datetime(df['epoch'].astype(str), errors='coerce', unit='ms')
df['timestamp'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()


# In[ ]:


print(df.columns.tolist())


# In[ ]:


# List of derived columns you want to keep
derived_cols = [
    "level_category",
    "component_category",
    "message_length",
    "hour",
    "day_of_week",
    "time_interval",
    "logs_15min_count",
    "warn_15min_count",
    "msg_len_mean_10",
    "msg_len_std_10",
    "text_feature",
    "peak_hours"
]

# Filter dataframe to contain only those columns
df = df[derived_cols]

# Show the dataframe
df.head()


# In[ ]:


# Step 1: Read the uploaded file
filename = next(iter(uploaded))  # Get the name of the uploaded file
with open(filename, 'r') as f:
    raw_lines = f.readlines()

# Step 2: Preprocess logs (clustering + embedding)
df, X_bert = preprocess_logs(raw_lines, n_clusters=20)

df = enrich_log_dataframe(df)

# Step 3: Classify logs (zero-shot + cluster-based labeling)
df = classify_logs(df)

# Step 4 (Optional): Visualize clusters using TSNE
visualize_clusters(X_bert, df)

# Step 5: Final output (core columns of interest)
df[['level', 'component', 'message', 'final_label', 'final_label_desc']].head()


# In[ ]:


df


# In[ ]:


import pandas as pd

def extract_warning_sequences(df, level_col="level", level_value="warning", window=99):
    """
    Extract sequences from df where rows contain `level_value` in `level_col`
    + the `window` rows before each match.
    """
    # Ensure df is sorted by timestamp index
    df = df.sort_index()

    # Find indices where condition is true
    warn_idx = df.index[df[level_col] == level_value]

    selected_idx = set()
    for idx in warn_idx:
        # Get integer position of this index
        pos = df.index.get_loc(idx)

        # Handle case where multiple rows have the same timestamp (get_loc can return a slice)
        if isinstance(pos, slice):
            pos = range(pos.start, pos.stop)
        else:
            pos = [pos]

        for p in pos:
            start = max(0, p - window)
            end = p + 1  # include warning row
            selected_idx.update(df.index[start:end])

    # Keep only the selected rows in order
    selected_df = df.loc[sorted(selected_idx)].copy()
    return selected_df


# Example usage
# new_df = extract_warning_sequences(df)


# In[ ]:


new_df = extract_warning_sequences(df)


# In[ ]:


new_df


# In[ ]:


# Save your dataset to CSV
new_df.to_csv("weblogic_logs_prepared.csv", index=False, encoding="utf-8")

print("✅ new_df saved to weblogic_logs_prepared.csv")


# In[ ]:


from google.colab import files
files.download("weblogic_logs_prepared.csv")


# In[ ]:


print(new_df.columns.tolist())


# In[ ]:


import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# In[ ]:


# Suppose your DataFrame is called df
filtered_df = df[df["level"] == "warning"]

# Display the rows
print(filtered_df)

# If you want to see the number of rows
print("Total rows with label 'log file rotated':", len(filtered_df))


# In[ ]:


filtered_df


# In[ ]:


# filter only rows with level == "warning"
warnings_df = df[df["level"] == "warning"]

# get all unique final_label values of those warnings
warning_labels = warnings_df["final_label"].unique()

print("All final_label values where level == 'warning':")
print(warning_labels)

