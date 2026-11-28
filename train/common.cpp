#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <thread>

#include <unistd.h>

namespace polypack::train {

// ---- Checkpoints --------------------------------------------------------------

ModelConfig modelConfigFor(const PackingEnv& env, int channels) {
  ModelConfig c;
  c.slots = env.slots();
  c.rotations = env.rotations();
  c.grid = env.grid();
  c.rasterSize = env.rasterSize();
  c.slotFeatures = PackingEnv::kSlotFeatures;
  c.globalFeatures = PackingEnv::kGlobalFeatures;
  c.criticFeatures = PackingEnv::kCriticFeatures;
  c.channels = channels;
  return c;
}

namespace {

constexpr int64_t kFormatVersion = 3;  // 2 = free placement, 3 = score is bins / offline bins

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
int64_t tryGetInt(torch::serialize::InputArchive& a, const std::string& key, int64_t fallback) {
  torch::Tensor t;
  return a.try_read("meta." + key, t) ? t.item<int64_t>() : fallback;
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
  putInt(archive, "env.grid", meta.env.grid);
  putInt(archive, "env.settle", meta.env.settle ? 1 : 0);
  putInt(archive, "env.rasterSize", meta.env.rasterSize);
  putInt(archive, "env.itemsPerEpisode", meta.env.itemsPerEpisode);
  putInt(archive, "env.minItemsPerEpisode", meta.env.minItemsPerEpisode);
  putInt(archive, "gen.minVertices", meta.env.generator.minVertices);
  putInt(archive, "gen.maxVertices", meta.env.generator.maxVertices);
  putReal(archive, "gen.minRadius", meta.env.generator.minRadius);
  putReal(archive, "gen.maxRadius", meta.env.generator.maxRadius);
  putReal(archive, "gen.innerRatio", meta.env.generator.innerRatio);
  putInt(archive, "gen.gridResolution", meta.env.generator.gridResolution);
  putInt(archive, "model.slots", meta.model.slots);
  putInt(archive, "model.rotations", meta.model.rotations);
  putInt(archive, "model.grid", meta.model.grid);
  putInt(archive, "model.rasterSize", meta.model.rasterSize);
  putInt(archive, "model.slotFeatures", meta.model.slotFeatures);
  putInt(archive, "model.globalFeatures", meta.model.globalFeatures);
  putInt(archive, "model.criticFeatures", meta.model.criticFeatures);
  putInt(archive, "model.channels", meta.model.channels);
  putInt(archive, "iteration", meta.iteration);
  putInt(archive, "envSteps", meta.envSteps);
  putReal(archive, "bestEvalScore", meta.bestEvalScore);

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
  archive.load_from(path, torch::Device(torch::kCPU));  // the file may hold CUDA tensors
  const int64_t format = getInt(archive, "format");
  if (format != kFormatVersion && format != 2) {
    throw std::runtime_error("unsupported checkpoint format in " + path +
                             " (checkpoints from the old top-drop version cannot be loaded)");
  }
  CheckpointMeta m;
  m.env.openBins = static_cast<int>(getInt(archive, "env.openBins"));
  m.env.grid = static_cast<int>(getInt(archive, "env.grid"));
  m.env.settle = getInt(archive, "env.settle") != 0;
  m.env.rasterSize = static_cast<int>(getInt(archive, "env.rasterSize"));
  m.env.itemsPerEpisode = static_cast<int>(getInt(archive, "env.itemsPerEpisode"));
  // Older checkpoints always had exactly itemsPerEpisode items.
  m.env.minItemsPerEpisode = static_cast<int>(tryGetInt(archive, "env.minItemsPerEpisode", 0));
  m.env.generator.minVertices = static_cast<int>(getInt(archive, "gen.minVertices"));
  m.env.generator.maxVertices = static_cast<int>(getInt(archive, "gen.maxVertices"));
  m.env.generator.minRadius = getReal(archive, "gen.minRadius");
  m.env.generator.maxRadius = getReal(archive, "gen.maxRadius");
  m.env.generator.innerRatio = getReal(archive, "gen.innerRatio");
  m.env.generator.gridResolution = static_cast<int>(getInt(archive, "gen.gridResolution"));
  m.model.slots = static_cast<int>(getInt(archive, "model.slots"));
  m.model.rotations = static_cast<int>(getInt(archive, "model.rotations"));
  m.model.grid = static_cast<int>(getInt(archive, "model.grid"));
  m.model.rasterSize = static_cast<int>(getInt(archive, "model.rasterSize"));
  m.model.slotFeatures = static_cast<int>(getInt(archive, "model.slotFeatures"));
  m.model.globalFeatures = static_cast<int>(getInt(archive, "model.globalFeatures"));
  m.model.criticFeatures = static_cast<int>(tryGetInt(archive, "model.criticFeatures", 0));
  m.model.channels = static_cast<int>(getInt(archive, "model.channels"));
  m.iteration = getInt(archive, "iteration");
  m.envSteps = getInt(archive, "envSteps");
  // Format 2 stored the best mean bin count, which is not comparable with the
  // new score: start the best-score tracking afresh.
  m.bestEvalScore = format == 2 ? 1e30 : getReal(archive, "bestEvalScore");
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

// ---- Offline reference cache ---------------------------------------------------

namespace {

// Bump when the offline packer changes in a way that changes its results.
constexpr int kOfflineVersion = 1;

std::string offlineKey(const EnvConfig& c, int patterns, std::uint64_t seed) {
  const GeneratorConfig& g = c.generator;
  std::ostringstream o;
  o << std::setprecision(17) << "v" << kOfflineVersion << " grid" << c.grid << " items" << c.minItemsPerEpisode
    << "-" << c.itemsPerEpisode << " verts" << g.minVertices << "-" << g.maxVertices << " radius" << g.minRadius
    << "-" << g.maxRadius << " inner" << g.innerRatio << " res" << g.gridResolution << " patterns" << patterns
    << " seed" << seed;
  return o.str();
}

}  // namespace

std::vector<OfflineResult> offlineReference(const EnvConfig& config, std::uint64_t seedBase, int count, int patterns,
                                            int threads, const std::string& cachePath) {
  std::unordered_map<std::string, OfflineResult> cached;
  if (!cachePath.empty()) {
    std::ifstream in(cachePath);
    std::string line;
    while (std::getline(in, line)) {
      std::istringstream fields(line);
      std::string key, bins, bound, pattern;
      if (!std::getline(fields, key, '\t') || !std::getline(fields, bins, '\t') ||
          !std::getline(fields, bound, '\t') || !std::getline(fields, pattern, '\t')) {
        continue;
      }
      OfflineResult r;
      r.bins = std::stoi(bins);
      r.lowerBound = std::stoi(bound);
      r.pattern = pattern;
      cached[key] = r;
    }
  }

  std::vector<OfflineResult> results(static_cast<std::size_t>(count));
  std::vector<int> missing;
  for (int i = 0; i < count; ++i) {
    const auto it = cached.find(offlineKey(config, patterns, seedBase + static_cast<std::uint64_t>(i)));
    if (it != cached.end()) {
      results[static_cast<std::size_t>(i)] = it->second;
    } else {
      missing.push_back(i);
    }
  }
  if (missing.empty()) return results;
  if (!cachePath.empty() && missing.size() < static_cast<std::size_t>(count)) {
    std::cout << "(" << count - static_cast<int>(missing.size()) << " of " << count << " from " << cachePath << ")"
              << std::endl;
  }

  std::vector<std::vector<Polygon>> episodes;
  for (int i : missing) episodes.push_back(episodeItems(config, seedBase + static_cast<std::uint64_t>(i)));
  // Progress counts pieces placed over all (episode, pattern) runs; a pattern
  // skipped because an earlier one already reached the lower bound counts as done.
  double totalPieces = 0;
  for (const auto& items : episodes) totalPieces += static_cast<double>(items.size()) * patterns;
  Progress progress("offline reference (" + std::to_string(missing.size()) + " episodes x " +
                        std::to_string(patterns) + " patterns)",
                    totalPieces);
  OfflineObserver observer;
  observer.onPlace = [&](int, int, const OfflinePlacement&) { progress.add(1); };
  observer.onSkip = [&](int episode, int) {
    progress.add(static_cast<double>(episodes[static_cast<std::size_t>(episode)].size()));
  };
  const std::vector<OfflineResult> computed =
      packOfflineEpisodes(episodes, config, patterns, threads, true, &observer);
  progress.finish();
  std::ofstream out;
  if (!cachePath.empty()) {
    const std::filesystem::path p(cachePath);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    out.open(cachePath, std::ios::app);
  }
  for (std::size_t k = 0; k < missing.size(); ++k) {
    const int i = missing[k];
    results[static_cast<std::size_t>(i)] = computed[k];
    if (out) {
      out << offlineKey(config, patterns, seedBase + static_cast<std::uint64_t>(i)) << '\t' << computed[k].bins << '\t'
          << computed[k].lowerBound << '\t' << computed[k].pattern << '\n';
    }
  }
  return results;
}

// ---- Observations -------------------------------------------------------------

ObsBatch ObsBatch::allocate(const PackingEnv& like, int64_t n) {
  const int64_t r = like.rasterSize();
  ObsBatch b;
  b.bins = torch::zeros({n, like.slots(), r, r}, torch::kUInt8);
  b.items = torch::zeros({n, like.rotations(), r, r}, torch::kUInt8);
  b.contact = torch::zeros({n, like.slots() * like.rotations(), r, r}, torch::kUInt8);
  b.slotF = torch::zeros({n, like.slots(), PackingEnv::kSlotFeatures}, torch::kFloat);
  b.globalF = torch::zeros({n, PackingEnv::kGlobalFeatures}, torch::kFloat);
  b.mask = torch::zeros({n, like.actionCount()}, torch::kBool);
  b.critic = torch::zeros({n, PackingEnv::kCriticFeatures}, torch::kFloat);
  return b;
}

ObsBatch ObsBatch::to(const torch::Device& device) const {
  return {bins.to(device), items.to(device), contact.to(device), slotF.to(device), globalF.to(device),
          mask.to(device), critic.to(device)};
}

ObsBatch ObsBatch::stack(const std::vector<ObsBatch>& steps) {
  std::vector<torch::Tensor> bins, items, contact, slotF, globalF, mask, critic;
  for (const ObsBatch& s : steps) {
    bins.push_back(s.bins);
    items.push_back(s.items);
    contact.push_back(s.contact);
    slotF.push_back(s.slotF);
    globalF.push_back(s.globalF);
    mask.push_back(s.mask);
    critic.push_back(s.critic);
  }
  return {torch::cat(bins), torch::cat(items), torch::cat(contact), torch::cat(slotF), torch::cat(globalF),
          torch::cat(mask), torch::cat(critic)};
}

ObsBatch ObsBatch::index(const torch::Tensor& idx) const {
  return {bins.index_select(0, idx), items.index_select(0, idx), contact.index_select(0, idx),
          slotF.index_select(0, idx), globalF.index_select(0, idx), mask.index_select(0, idx),
          critic.index_select(0, idx)};
}

void observeInto(const PackingEnv& env, ObsBatch& out, int64_t i) {
  // bool and uint8 share a byte layout.
  env.observe(out.bins.data_ptr<uint8_t>() + i * out.bins.stride(0),
              out.items.data_ptr<uint8_t>() + i * out.items.stride(0),
              out.contact.data_ptr<uint8_t>() + i * out.contact.stride(0),
              out.slotF.data_ptr<float>() + i * out.slotF.stride(0),
              out.globalF.data_ptr<float>() + i * out.globalF.stride(0),
              reinterpret_cast<uint8_t*>(out.mask.data_ptr<bool>()) + i * out.mask.stride(0));
  env.criticFeatures(out.critic.data_ptr<float>() + i * out.critic.stride(0));
}

void observeAll(const std::vector<PackingEnv>& envs, ObsBatch& out, int threads) {
  parallelFor(static_cast<int>(envs.size()), threads,
              [&](int i) { observeInto(envs[static_cast<std::size_t>(i)], out, i); });
}

std::vector<int> unfinished(const std::vector<PackingEnv>& envs) {
  std::vector<int> result;
  for (std::size_t i = 0; i < envs.size(); ++i) {
    if (!envs[i].done()) result.push_back(static_cast<int>(i));
  }
  return result;
}

ObsBatch observeSome(const std::vector<PackingEnv>& envs, const std::vector<int>& indices, int threads) {
  ObsBatch out = ObsBatch::allocate(envs.at(0), static_cast<int64_t>(indices.size()));
  parallelFor(static_cast<int>(indices.size()), threads, [&](int k) {
    observeInto(envs[static_cast<std::size_t>(indices[static_cast<std::size_t>(k)])], out, k);
  });
  return out;
}

PolicyOutput runModel(PolicyNet& model, const ObsBatch& b) {
  // Older models take fewer global features (the first ones keep their
  // meaning) and no critic features.
  const ModelConfig& c = model->config();
  return model->forward(b.bins, b.items, b.contact, b.slotF, b.globalF.narrow(1, 0, c.globalFeatures), b.mask,
                        b.critic.narrow(1, 0, c.criticFeatures));
}

// ---- Progress -----------------------------------------------------------------

std::string formatDuration(double seconds) {
  if (!(seconds >= 0) || seconds > 1e8) return "?";
  const long s = static_cast<long>(seconds + 0.5);
  char buf[32];
  if (s < 60) std::snprintf(buf, sizeof buf, "%lds", s);
  else if (s < 3600) std::snprintf(buf, sizeof buf, "%ldm%02lds", s / 60, s % 60);
  else if (s < 86400) std::snprintf(buf, sizeof buf, "%ldh%02ldm", s / 3600, (s / 60) % 60);
  else std::snprintf(buf, sizeof buf, "%ldd%02ldh", s / 86400, (s / 3600) % 24);
  return buf;
}

Progress::Progress(std::string label, double total)
    : label_(std::move(label)),
      total_(total),
      start_(std::chrono::steady_clock::now()),
      lastDraw_(start_ - std::chrono::hours(1)),
      interval_(isatty(STDERR_FILENO) ? 0.25 : 2.0) {}

Progress::~Progress() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!finished_) erase();
}

void Progress::add(double n) {
  const std::lock_guard<std::mutex> lock(mutex_);
  done_ += n;
  draw(false);
}

void Progress::set(double done) {
  const std::lock_guard<std::mutex> lock(mutex_);
  done_ = done;
  draw(false);
}

void Progress::setDetail(std::string detail) {
  const std::lock_guard<std::mutex> lock(mutex_);
  detail_ = std::move(detail);
}

void Progress::clear() {
  const std::lock_guard<std::mutex> lock(mutex_);
  erase();
  lastDraw_ = std::chrono::steady_clock::now() - std::chrono::hours(1);  // redraw on the next update
}

void Progress::finish(bool keepLine) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (finished_) return;
  finished_ = true;
  if (!keepLine) {
    erase();
    return;
  }
  done_ = total_;
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  std::ostringstream line;
  line << label_ << ": done in " << formatDuration(elapsed);
  if (!detail_.empty()) line << "  " << detail_;
  std::cout.flush();
  std::string text = line.str();
  const std::size_t len = text.size();
  if (len < shown_) text += std::string(shown_ - len, ' ');
  std::cerr << '\r' << text << '\n' << std::flush;
  shown_ = 0;
}

void Progress::erase() {
  if (shown_ == 0) return;
  std::cerr << '\r' << std::string(shown_, ' ') << '\r' << std::flush;
  shown_ = 0;
}

void Progress::draw(bool force) {
  if (finished_) return;
  const auto now = std::chrono::steady_clock::now();
  if (!force && std::chrono::duration<double>(now - lastDraw_).count() < interval_) return;
  lastDraw_ = now;
  const double elapsed = std::chrono::duration<double>(now - start_).count();
  const double fraction = total_ > 0 ? std::clamp(done_ / total_, 0.0, 1.0) : 0.0;
  constexpr int kWidth = 30;
  const int filled = static_cast<int>(fraction * kWidth);
  std::ostringstream line;
  line << label_ << " [" << std::string(static_cast<std::size_t>(filled), '=')
       << (filled < kWidth ? ">" : "") << std::string(static_cast<std::size_t>(std::max(0, kWidth - filled - 1)), ' ')
       << "] " << std::fixed << std::setprecision(1) << std::setw(5) << 100.0 * fraction << "%  ";
  if (total_ >= 2 && std::abs(total_ - std::round(total_)) < 1e-9 && std::abs(done_ - std::round(done_)) < 1e-9) {
    line << static_cast<long long>(std::llround(done_)) << '/' << static_cast<long long>(std::llround(total_)) << "  ";
  }
  line << formatDuration(elapsed) << " elapsed  ETA "
       << (fraction > 0 ? formatDuration(elapsed * (1.0 - fraction) / fraction) : std::string("?"));
  if (!detail_.empty()) line << "  " << detail_;
  std::string text = line.str();
  const std::size_t len = text.size();
  if (len < shown_) text += std::string(shown_ - len, ' ');
  std::cout.flush();  // keep normal output and the bar in order
  std::cerr << '\r' << text << std::flush;
  shown_ = len;
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
