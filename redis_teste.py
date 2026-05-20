import redis

r = redis.Redis(
    host='scissors-increase-rain-90411.db.redis.io',
    port=11893,
    decode_responses=True,
    username='default',
    password='hjbIbk8Lk652OqoCkHmc3YxPHNGCOD1m',
)

success = r.set('foo', 'bar')
# True

result = r.get('foo')
print(result)
# >>> bar
