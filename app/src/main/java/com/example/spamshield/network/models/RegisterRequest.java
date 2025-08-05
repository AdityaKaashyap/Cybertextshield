// RegisterRequest.java
// RegisterRequest.java
package com.example.spamshield.network.models;

public class RegisterRequest {
    private String username;
    private String country_code;
    private String phone_number;

    public RegisterRequest(String username, String country_code, String phone_number) {
        this.username = username;
        this.country_code = country_code;
        this.phone_number = phone_number;
    }
}

// RegisterResponse.java
package com.example.spamshield.network.models;

public class RegisterResponse {
    private String user_id;
    private String message;

    public String getUser_id() {
        return user_id;
    }

    public String getMessage() {
        return message;
    }
}
