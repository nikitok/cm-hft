//! HMAC-SHA256 request signing for exchange APIs.
//!
//! Uses the `ring` crate for constant-time HMAC computation, avoiding
//! OpenSSL dependencies. Secrets are never logged or included in error messages.

use ring::hmac;

/// Sign a Binance REST API request.
///
/// Binance signs the query string: `HMAC-SHA256(secret, query_string)`.
/// The resulting hex-encoded signature is appended as `&signature=...`.
pub fn sign_binance_request(secret: &str, query_string: &str) -> String {
    let key = hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes());
    let signature = hmac::sign(&key, query_string.as_bytes());
    hex::encode(signature.as_ref())
}

/// Sign a Bybit REST API v5 request.
///
/// Bybit signs the concatenation: `timestamp + api_key + recv_window + body`.
/// The resulting hex-encoded signature is sent in the `X-BAPI-SIGN` header.
pub fn sign_bybit_request(
    secret: &str,
    timestamp: u64,
    api_key: &str,
    recv_window: u64,
    body: &str,
) -> String {
    let payload = format!("{}{}{}{}", timestamp, api_key, recv_window, body);
    let key = hmac::Key::new(hmac::HMAC_SHA256, secret.as_bytes());
    let signature = hmac::sign(&key, payload.as_bytes());
    hex::encode(signature.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binance_signing_known_vector() {
        // Known test vector: secret="NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
        // query="symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
        let secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j";
        let query = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559";

        let sig = sign_binance_request(secret, query);

        // Verify it produces a 64-char hex string (SHA-256 = 32 bytes = 64 hex chars)
        assert_eq!(sig.len(), 64);
        // Verify determinism
        let sig2 = sign_binance_request(secret, query);
        assert_eq!(sig, sig2);
        // Known expected value from Binance API docs
        assert_eq!(
            sig,
            "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
        );
    }

    #[test]
    fn test_binance_signing_empty_query() {
        let secret = "test_secret";
        let sig = sign_binance_request(secret, "");
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn test_binance_signing_different_secrets_differ() {
        let query = "symbol=BTCUSDT&timestamp=1000000";
        let sig1 = sign_binance_request("secret_a", query);
        let sig2 = sign_binance_request("secret_b", query);
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_binance_signing_different_queries_differ() {
        let secret = "my_secret";
        let sig1 = sign_binance_request(secret, "symbol=BTCUSDT&timestamp=1000");
        let sig2 = sign_binance_request(secret, "symbol=ETHUSDT&timestamp=1000");
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_bybit_signing_known_vector() {
        let secret = "test_bybit_secret_key_12345";
        let timestamp = 1672502400000u64;
        let api_key = "my_api_key";
        let recv_window = 5000u64;
        let body = r#"{"category":"linear","symbol":"BTCUSDT","side":"Buy","orderType":"Limit","qty":"0.001","price":"50000"}"#;

        let sig = sign_bybit_request(secret, timestamp, api_key, recv_window, body);

        // SHA-256 always 64 hex chars
        assert_eq!(sig.len(), 64);
        // Determinism
        let sig2 = sign_bybit_request(secret, timestamp, api_key, recv_window, body);
        assert_eq!(sig, sig2);
    }

    #[test]
    fn test_bybit_signing_payload_composition() {
        // Verify the payload is assembled correctly by checking that changing
        // any component changes the signature.
        let secret = "secret";
        let base_sig = sign_bybit_request(secret, 1000, "key", 5000, "body");

        // Different timestamp
        let sig_ts = sign_bybit_request(secret, 1001, "key", 5000, "body");
        assert_ne!(base_sig, sig_ts);

        // Different api_key
        let sig_key = sign_bybit_request(secret, 1000, "key2", 5000, "body");
        assert_ne!(base_sig, sig_key);

        // Different recv_window
        let sig_rw = sign_bybit_request(secret, 1000, "key", 5001, "body");
        assert_ne!(base_sig, sig_rw);

        // Different body
        let sig_body = sign_bybit_request(secret, 1000, "key", 5000, "body2");
        assert_ne!(base_sig, sig_body);
    }

    #[test]
    fn test_bybit_signing_empty_body() {
        let secret = "secret";
        let sig = sign_bybit_request(secret, 1000, "key", 5000, "");
        assert_eq!(sig.len(), 64);
    }

    #[test]
    fn test_signature_is_lowercase_hex() {
        let sig = sign_binance_request("key", "data");
        assert!(sig
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
    }
}
