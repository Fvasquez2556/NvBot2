# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.9.x   | :white_check_mark: |
| < 0.9   | :x:                |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

### For Security Issues
Instead, please report them via email to: **security@nvbot2.com**

### What to Include
Please include the following information:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline
- **Initial Response**: Within 48 hours
- **Assessment**: Within 5 business days
- **Fix Timeline**: Critical issues within 14 days, others within 30 days
- **Disclosure**: Coordinated disclosure after fix is available

## Security Considerations for Trading Bots

### API Security
- **Never commit API keys** to version control
- Use environment variables for sensitive configuration
- Implement proper API key rotation procedures
- Monitor API usage for unusual patterns

### Risk Management
- Implement position sizing limits
- Set maximum daily/weekly loss limits  
- Use multiple layers of risk validation
- Monitor for unusual trading patterns

### Data Security
- Encrypt sensitive trading data at rest
- Use secure connections (TLS) for all API calls
- Implement proper access controls
- Regular security audits of data handling

### Infrastructure Security
- Keep dependencies updated
- Use container security scanning
- Implement proper logging and monitoring
- Regular backup and disaster recovery testing

## Security Best Practices

### For Contributors
1. **Code Review**: All code must be reviewed before merging
2. **Dependency Scanning**: Use tools like `safety` and `bandit`
3. **Static Analysis**: Run security linters on all code
4. **Testing**: Include security tests in test suites

### For Users
1. **Environment Isolation**: Run in isolated environments
2. **API Permissions**: Use read-only APIs when possible
3. **Monitor Logs**: Watch for unusual behavior
4. **Backup Strategies**: Maintain secure backups of configurations

## Known Security Considerations

### Trading Risks
- **Market Risk**: Automated trading can lead to significant losses
- **Execution Risk**: Network issues can cause missed opportunities
- **Model Risk**: ML models may behave unexpectedly in new market conditions

### Technical Risks
- **API Rate Limits**: Exceeding limits can cause account restrictions
- **Data Quality**: Poor data can lead to incorrect trading decisions
- **System Downtime**: Infrastructure failures can impact trading

## Security Tools Used

- **bandit**: Python security linter
- **safety**: Dependency vulnerability scanner
- **pytest-security**: Security testing framework
- **Docker security scanning**: Container vulnerability detection

## Disclosure Policy

We follow responsible disclosure practices:

1. **Private Report**: Issue reported privately to security team
2. **Investigation**: Security team investigates and confirms
3. **Fix Development**: Patch developed and tested
4. **Release**: Security update released to users
5. **Public Disclosure**: Issue disclosed publicly after fix deployment

## Security Updates

Security updates will be:
- Released as patch versions (e.g., 1.0.1)
- Announced in GitHub Security Advisories
- Detailed in CHANGELOG.md
- Communicated via email to registered users

## Contact

- **Security Email**: security@nvbot2.com
- **General Issues**: Use GitHub Issues for non-security bugs
- **Security Advisories**: Watch GitHub Security tab for updates

---

**Remember**: This project involves financial trading. Always use proper risk management and never trade with money you cannot afford to lose.
