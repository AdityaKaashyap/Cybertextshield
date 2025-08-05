// PredictRequest.java
// PredictRequest.java
package com.example.spamshield.network.models;

public class PredictRequest {
    private String text;
    private String user_id;

    public PredictRequest(String text, String user_id) {
        this.text = text;
        this.user_id = user_id;
    }
}

// PredictResponse.java
package com.example.spamshield.network.models;

import java.util.Map;

public class PredictResponse {
    private String message;
    private String prediction;
    private double confidence;
    private Map<String, Double> probabilities;

    public String getMessage() {
        return message;
    }

    public String getPrediction() {
        return prediction;
    }

    public double getConfidence() {
        return confidence;
    }

    public Map<String, Double> getProbabilities() {
        return probabilities;
    }
}
