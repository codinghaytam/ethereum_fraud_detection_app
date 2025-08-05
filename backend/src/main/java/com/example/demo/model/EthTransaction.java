package com.example.demo.model;

import lombok.Data;

@Data
public class EthTransaction {
    private String blockNumber;
    private String blockHash;
    private String timeStamp;
    private String hash;
    private String nonce;
    private String transactionIndex;
    private String from;
    private String to;
    private String value;
    private String gas;
    private String gasPrice;
    private String input;
    private String methodId;
    private String functionName;
    private String contractAddress;
    private String cumulativeGasUsed;
    private String txreceipt_status;
    private String gasUsed;
    private String confirmations;
    private String isError;
}
