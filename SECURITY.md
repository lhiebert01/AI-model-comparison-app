# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please send an email to [your-email@example.com]. All security vulnerabilities will be promptly addressed.

Please do not disclose security-related issues publicly until a fix has been announced.

## API Keys and Sensitive Data

This project requires API keys from:
- Google AI (Gemini)
- OpenAI

Never commit real API keys to the repository. Always use the `.env` file for local development and secure environment variables for production deployment.

## Best Practices

1. Never commit `.env` files containing real credentials
2. Use environment variables for all sensitive data
3. Regularly rotate API keys
4. Follow the principle of least privilege when setting up API access
5. Keep all dependencies updated to their latest secure versions

## Development Setup

1. Copy `.env.template` to `.env`
2. Replace placeholder values with real API keys
3. Ensure `.env` is listed in `.gitignore`
4. Verify no sensitive data is being tracked by git before pushing

## Production Deployment

For production deployment:
1. Use secure environment variable storage
2. Enable audit logging
3. Implement rate limiting
4. Monitor for unusual activity