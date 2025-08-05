package com.example.demo.model;

import lombok.Data;

import java.util.List;
@Data
public class Raison {
    private String title;
    private String description;
    private List<String> examples;

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public List<String> getExamples() {
        return examples;
    }

    public void setExamples(List<String> examples) {
        this.examples = examples;
    }

    public Raison(String title, String description, List<String> examples) {
        this.title = title;
        this.description = description;
        this.examples = examples;
    }
}
