#ifndef ODL_PROCESSOR_STORAGE_REDIS_FEATURE_STORE_H_
#define ODL_PROCESSOR_STORAGE_REDIS_FEATURE_STORE_H_
#include <vector>
#include <string>
#include <thread>

#include "odl_processor/storage/feature_store.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

class redisContext;
class redisAsyncContext;
struct event_base;

namespace tensorflow {
namespace processor {

class LocalRedis : public FeatureStore {
  public:
    struct Config {
      std::string ip;
      int32_t port = 0;
      std::string passwd;
    };

    LocalRedis(Config config);
    ~LocalRedis();

    Status Cleanup();

    Status BatchGet(uint64_t model_version,
                    uint64_t feature2id,
                    const char* const keys,
                    char* const values,
                    size_t bytes_per_key,
                    size_t bytes_per_values,
                    size_t N,
                    const char* default_value);

    Status BatchSet(uint64_t model_version,
                    uint64_t feature2id,
                    const char* const keys,
                    const char* const values,
                    size_t bytes_per_key,
                    size_t bytes_per_values,
                    size_t N);

    Status BatchGetAsync(uint64_t model_version,
                         uint64_t feature2id,
                         const char* const keys,
                         char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         const char* default_value,
                         BatchGetCallback cb);

    Status BatchSetAsync(uint64_t model_version,
                         uint64_t feature2id,
                         const char* const keys,
                         const char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         BatchSetCallback cb);

  private:
    std::string ip_;
    int32_t port_;
    redisContext *c_;
};

class ClusterRedis : public FeatureStore {
  public:
    ClusterRedis(const std::string& redis_url,
        const std::string& redis_password);

    Status Cleanup() {
      return errors::Unimplemented("[Redis] Unimplement Cleanup().");
    }

    Status BatchGet(uint64_t model_version,
                    uint64_t feature2id,
                    const char* const keys,
                    char* const values,
                    size_t bytes_per_key,
                    size_t bytes_per_values,
                    size_t N,
                    const char* default_value) {
      return errors::Unimplemented("[redis] unimplement batchget() in sync mode.");
    }

    Status BatchSet(uint64_t model_version,
                    uint64_t feature2id,
                    const char* const keys,
                    const char* const values,
                    size_t bytes_per_key,
                    size_t bytes_per_values,
                    size_t N) {
      return errors::Unimplemented("[redis] unimplement batchset() in sync mode.");
    }

    Status BatchGetAsync(uint64_t model_version,
                         uint64_t feature2id,
                         const char* const keys,
                         char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         const char* default_value,
                         BatchGetCallback cb) {
      return errors::Unimplemented("[Redis] Unimplement BatchGet() in async mode.");
    }

    Status BatchSetAsync(uint64_t model_version,
                         uint64_t feature2id,
                         const char* const keys,
                         const char* const values,
                         size_t bytes_per_key,
                         size_t bytes_per_values,
                         size_t N,
                         BatchSetCallback cb) {
      return errors::Unimplemented("[Redis] Unimplement BatchSet() in async mode.");
    }

};

} // namespace processor
} // namespace tensorflow

#endif // ODL_PROCESSOR_STORAGE_REDIS_FEATURE_STORE_H_
