#include "common.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <thread>

namespace polypack::train {

// ---- Checkpoints --------------------------------------------------------------

ModelConfig modelConfigFor(const PackingEnv& env, int channels) {
  ModelConfig c;
  c.slots = env.slots();
  c.rotations = env.rotations();
  c.columns = env.columns();
  c.rasterSize = env.rasterSize();
  c.slotFeatures = PackingEnv::kSlotFeatures;
  c.globalFeatures = PackingEnv::kGlobalFeatures;
  c.channels = channels;
  return c;
}

namespace {

constexpr int64_t kFormatVersion = 1;

void putInt(torch::serialize::OutputArchive& a, const std::string& key, int64_t v) {
  a.write("meta." + key, torch::tensor(v, torch::kInt64));
}
void putReal(torch::serialize::OutputArchive& a, const std::string& key, double v) {
  a.write("meta." + key, torch::tensor(v, torch::kFloat64));
}
int64_t getInt(torch::serialize::InputArchive& a, const std::string& key) {
  torch::Tensor t;
  a.read("meta." + key, t);
  return t.item<int64_t>();
}
double getReal(torch::serialize::InputArchive& a, const std::string& key) {
  torch::Tensor t;
  a.read("meta." + key, t);
  return t.item<double>();
}

}  // namespace

void saveCheckpoint(const std::string& path, PolicyNet& model, torch::optim::Optimizer* optimizer,
                    const CheckpointMeta& meta) {
  torch::serialize::OutputArchive archive;
  putInt(archive, "format", kFormatVersion);
  putInt(archive, "env.openBins", meta.env.openBins);
  putInt(archive, "env.columns", meta.env.columns);
  putInt(archive, "env.rasterSize", meta.env.rasterSize);
  putInt(archive, "env.itemsPerEpisode", meta.env.itemsPerEpisode);
  putInt(archive, "gen.minVertices", meta.env.generator.minVertices);
  putInt(archive, "gen.maxVertices", meta.env.generator.maxVertices);
  putReal(archive, "gen.minRadius", meta.env.generator.minRadius);
  putReal(archive, "gen.maxRadius", meta.env.generator.maxRadius);
  putReal(archive, "gen.innerRatio", meta.env.generator.innerRatio);
  putInt(archive, "gen.gridResolution", meta.env.generator.gridResolution);
  putInt(archive, "model.slots", meta.model.slots);
  putInt(archive, "model.rotations", meta.model.rotations);
  putInt(archive, "model.columns", meta.model.columns);
  putInt(archive, "model.rasterSize", meta.model.rasterSize);
  putInt(archive, "model.slotFeatures", meta.model.slotFeatures);
  putInt(archive, "model.globalFeatures", meta.model.globalFeatures);
  putInt(archive, "model.channels", meta.model.channels);
  putInt(archive, "iteration", meta.iteration);
  putInt(archive, "envSteps", meta.envSteps);
  putReal(archive, "bestEvalBins", meta.bestEvalBins);

  torch::serialize::OutputArchive weights;
  model->save(weights);
  archive.write("model", weights);
  if (optimizer != nullptr) {
    torch::serialize::OutputArchive opt;
    optimizer->save(opt);
    archive.write("optimizer", opt);
  }

  const std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  // Write to a temporary file first so an interrupted save never corrupts the model.
  const std::string tmp = path + ".tmp";
  archive.save_to(tmp);
  std::filesystem::rename(tmp, path);
}

CheckpointMeta loadCheckpointMeta(const std::string& path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  if (getInt(archive, "format") != kFormatVersion) {
    throw std::runtime_error("unsupported checkpoint format in " + path);
  }
  CheckpointMeta m;
  m.env.openBins = static_cast<int>(getInt(archive, "env.openBins"));
  m.env.columns = static_cast<int>(getInt(archive, "env.columns"));
  m.env.rasterSize = static_cast<int>(getInt(archive, "env.rasterSize"));
  m.env.itemsPerEpisode = static_cast<int>(getInt(archive, "env.itemsPerEpisode"));
  m.env.generator.minVertices = static_cast<int>(getInt(archive, "gen.minVertices"));
  m.env.generator.maxVertices = static_cast<int>(getInt(archive, "gen.maxVertices"));
  m.env.generator.minRadius = getReal(archive, "gen.minRadius");
  m.env.generator.maxRadius = getReal(archive, "gen.maxRadius");
  m.env.generator.innerRatio = getReal(archive, "gen.innerRatio");
  m.env.generator.gridResolution = static_cast<int>(getInt(archive, "gen.gridResolution"));
  m.model.slots = static_cast<int>(getInt(archive, "model.slots"));
  m.model.rotations = static_cast<int>(getInt(archive, "model.rotations"));
  m.model.columns = static_cast<int>(getInt(archive, "model.columns"));
  m.model.rasterSize = static_cast<int>(getInt(archive, "model.rasterSize"));
  m.model.slotFeatures = static_cast<int>(getInt(archive, "model.slotFeatures"));
  m.model.globalFeatures = static_cast<int>(getInt(archive, "model.globalFeatures"));
  m.model.channels = static_cast<int>(getInt(archive, "model.channels"));
  m.iteration = getInt(archive, "iteration");
  m.envSteps = getInt(archive, "envSteps");
  m.bestEvalBins = getReal(archive, "bestEvalBins");
  return m;
}

void loadCheckpointWeights(const std::string& path, PolicyNet& model,
                           torch::optim::Optimizer* optimizer, const torch::Device& device) {
  // Map every tensor onto `device`, so a GPU-trained model loads on a CPU-only
  // machine and vice versa.
  torch::serialize::InputArchive archive;
  archive.load_from(path, device);
  torch::serialize::InputArchive weights;
  archive.read("model", weights);
  model->load(weights);
  if (optimizer != nullptr) {
    torch::serialize::InputArchive opt;
    if (archive.try_read("optimizer", opt)) optimizer->load(opt);
  }
}

// ---- Observations -------------------------------------------------------------

ObsBatch ObsBatch::allocate(const PackingEnv& like, int64_t n) {
  const int64_t r = like.rasterSize();
  ObsBatch b;
  b.bins = torch::zeros({n, like.slots(), r, r}, torch::kUInt8);
  b.items = torch::zeros({n, like.rotations(), r, r}, torch::kUInt8);
  b.slotF = torch::zeros({n, like.slots(), PackingEnv::kSlotFeatures}, torch::kFloat);
  b.globalF = torch::zeros({n, PackingEnv::kGlobalFeatures}, torch::kFloat);
  b.mask = torch::zeros({n, like.actionCount()}, torch::kBool);
  return b;
}

ObsBatch ObsBatch::to(const torch::Device& device) const {
  return {bins.to(device), items.to(device), slotF.to(device), globalF.to(device), mask.to(device)};
}

ObsBatch ObsBatch::stack(const std::vector<ObsBatch>& steps) {
  std::vector<torch::Tensor> bins, items, slotF, globalF, mask;
  for (const ObsBatch& s : steps) {
    bins.push_back(s.bins);
    items.push_back(s.items);
    slotF.push_back(s.slotF);
    globalF.push_back(s.globalF);
    mask.push_back(s.mask);
  }
  return {torch::cat(bins), torch::cat(items), torch::cat(slotF), torch::cat(globalF), torch::cat(mask)};
}

ObsBatch ObsBatch::index(const torch::Tensor& idx) const {
  return {bins.index_select(0, idx), items.index_select(0, idx), slotF.index_select(0, idx),
          globalF.index_select(0, idx), mask.index_select(0, idx)};
}

void observeAll(const std::vector<PackingEnv>& envs, ObsBatch& out, int threads) {
  const auto n = static_cast<int>(envs.size());
  const int64_t binStride = out.bins.stride(0);
  const int64_t itemStride = out.items.stride(0);
  const int64_t slotStride = out.slotF.stride(0);
  const int64_t globalStride = out.globalF.stride(0);
  const int64_t maskStride = out.mask.stride(0);
  auto* bins = out.bins.data_ptr<uint8_t>();
  auto* items = out.items.data_ptr<uint8_t>();
  auto* slotF = out.slotF.data_ptr<float>();
  auto* globalF = out.globalF.data_ptr<float>();
  // bool and uint8 share a byte layout.
  auto* mask = reinterpret_cast<uint8_t*>(out.mask.data_ptr<bool>());
  parallelFor(n, threads, [&](int i) {
    envs[static_cast<std::size_t>(i)].observe(bins + i * binStride, items + i * itemStride,
                                               slotF + i * slotStride, globalF + i * globalStride,
                                               mask + i * maskStride);
  });
}

// ---- Misc ---------------------------------------------------------------------

void parallelFor(int n, int threads, const std::function<void(int)>& fn) {
  const int workers = std::max(1, std::min(threads, n));
  if (workers == 1) {
    for (int i = 0; i < n; ++i) fn(i);
    return;
  }
  std::vector<std::thread> pool;
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(workers));
  for (int w = 0; w < workers; ++w) {
    pool.emplace_back([&, w] {
      try {
        for (int i = w; i < n; i += workers) fn(i);
      } catch (...) {
        errors[static_cast<std::size_t>(w)] = std::current_exception();
      }
    });
  }
  for (std::thread& t : pool) t.join();
  for (const std::exception_ptr& e : errors) {
    if (e) std::rethrow_exception(e);
  }
}

torch::Device pickDevice(const std::string& name) {
  if (name == "auto") return torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  torch::Device device(name);
  if (device.is_cuda() && !torch::cuda::is_available()) {
    throw std::runtime_error("CUDA requested but not available (is this a CUDA build of libtorch?)");
  }
  return device;
}

Args::Args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + key);
    key = key.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      values_[key] = argv[++i];
    } else {
      values_[key] = "1";  // bare flag
    }
  }
}

std::string Args::str(const std::string& key, const std::string& fallback) const {
  used_[key] = true;
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : it->second;
}

int64_t Args::integer(const std::string& key, int64_t fallback) const {
  const std::string v = str(key, "");
  return v.empty() ? fallback : std::stoll(v);
}

double Args::real(const std::string& key, double fallback) const {
  const std::string v = str(key, "");
  return v.empty() ? fallback : std::stod(v);
}

void Args::rejectUnknown() const {
  for (const auto& [key, value] : values_) {
    if (!used_.count(key)) throw std::invalid_argument("unknown option --" + key);
  }
}

}  // namespace polypack::train
