# Troubleshooting & Operational Recovery

## 1. Common Issues & Resolutions

### "Index is not loaded or ready for search" (503 / 404)
- **Cause**: No index version has been built or activated in `data/indexes/`.
- **Fix**: Run `python scripts/build_index.py` or trigger `/api/v1/index/rebuild`. On initial startup, the service auto-creates an initial version `v001`.

### "Embedding dimension mismatch: expected 512, got X"
- **Cause**: A custom embedding model was configured whose raw dimension is not 512.
- **Fix**: The platform automatically generates a deterministic orthogonal projection matrix for mismatched models. Ensure `EMBEDDING_DIMENSION=512` is set in configuration.

### "Rate limit exceeded" (429)
- **Cause**: Client sent more requests than `RATE_LIMIT_PER_MINUTE`.
- **Fix**: Increase `RATE_LIMIT_PER_MINUTE` in `.env` or adjust client concurrency.

### "Tenant access denied" (403)
- **Cause**: Client attempted to read or delete a document belonging to a different tenant.
- **Fix**: Supply matching `tenant_id` query parameter.

## 2. Backup & Disaster Recovery Procedures
1. **Backup Active Index**:
   ```bash
   tar -czvf backup_index_$(date +%Y%m%d).tar.gz data/indexes/current/
   ```
2. **Restore Index**:
   ```bash
   mkdir -p data/indexes/v_restored
   tar -xzvf backup_index_*.tar.gz -C data/indexes/v_restored/
   python -m app.cli validate-index --version v_restored
   ```
3. **Activate Restored Index**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/index/activate \
     -H "X-API-Key: admin-secret-key-12345" \
     -H "Content-Type: application/json" \
     -d '{"version_name": "v_restored"}'
   ```
4. **Emergency Rollback**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/index/rollback \
     -H "X-API-Key: admin-secret-key-12345"
   ```
