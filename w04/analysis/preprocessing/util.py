def malicious_translation(string):
    translations = {
        'benign': 0,
        'phishing': 1,
        'defacement': 2,
        'malware': 3
    }
    return translations[string]