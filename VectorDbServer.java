import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class VectorDbServer {
    private static final int DIMS = 16;

    @FunctionalInterface
    interface DistFn {
        float apply(List<Float> a, List<Float> b);
    }

    static final class VectorItem {
        final int id;
        final String metadata;
        final String category;
        final List<Float> emb;

        VectorItem(int id, String metadata, String category, List<Float> emb) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.emb = emb;
        }
    }

    static final class DocItem {
        final int id;
        final String title;
        final String text;
        final List<Float> emb;

        DocItem(int id, String title, String text, List<Float> emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    static final class Pair {
        final float dist;
        final int id;

        Pair(float dist, int id) {
            this.dist = dist;
            this.id = id;
        }
    }

    static final class Distance {
        static float euclidean(List<Float> a, List<Float> b) {
            float s = 0f;
            for (int i = 0; i < a.size(); i++) {
                float d = a.get(i) - b.get(i);
                s += d * d;
            }
            return (float) Math.sqrt(s);
        }

        static float cosine(List<Float> a, List<Float> b) {
            float dot = 0f;
            float na = 0f;
            float nb = 0f;
            for (int i = 0; i < a.size(); i++) {
                float av = a.get(i);
                float bv = b.get(i);
                dot += av * bv;
                na += av * av;
                nb += bv * bv;
            }
            if (na < 1e-9f || nb < 1e-9f) {
                return 1.0f;
            }
            return 1.0f - (float) (dot / (Math.sqrt(na) * Math.sqrt(nb)));
        }

        static float manhattan(List<Float> a, List<Float> b) {
            float s = 0f;
            for (int i = 0; i < a.size(); i++) {
                s += Math.abs(a.get(i) - b.get(i));
            }
            return s;
        }

        static DistFn get(String metric) {
            if ("cosine".equals(metric)) {
                return Distance::cosine;
            }
            if ("manhattan".equals(metric)) {
                return Distance::manhattan;
            }
            return Distance::euclidean;
        }
    }

    static final class BruteForce {
        private final List<VectorItem> items = new ArrayList<>();

        void insert(VectorItem item) {
            items.add(item);
        }

        void remove(int id) {
            items.removeIf(v -> v.id == id);
        }

        List<Pair> knn(List<Float> q, int k, DistFn distFn) {
            List<Pair> out = new ArrayList<>(items.size());
            for (VectorItem item : items) {
                out.add(new Pair(distFn.apply(q, item.emb), item.id));
            }
            out.sort(Comparator.comparingDouble(a -> a.dist));
            if (out.size() > k) {
                return new ArrayList<>(out.subList(0, k));
            }
            return out;
        }
    }

    static final class KDTree {
        static final class Node {
            final VectorItem item;
            Node left;
            Node right;

            Node(VectorItem item) {
                this.item = item;
            }
        }

        private Node root;
        private final int dims;

        KDTree(int dims) {
            this.dims = dims;
        }

        void insert(VectorItem item) {
            root = insertRec(root, item, 0);
        }

        void rebuild(List<VectorItem> items) {
            root = null;
            for (VectorItem item : items) {
                insert(item);
            }
        }

        private Node insertRec(Node node, VectorItem item, int depth) {
            if (node == null) {
                return new Node(item);
            }
            int axis = depth % dims;
            if (item.emb.get(axis) < node.item.emb.get(axis)) {
                node.left = insertRec(node.left, item, depth + 1);
            } else {
                node.right = insertRec(node.right, item, depth + 1);
            }
            return node;
        }

        List<Pair> knn(List<Float> q, int k, DistFn distFn) {
            PriorityQueue<Pair> heap = new PriorityQueue<>(Comparator.comparingDouble(a -> -a.dist));
            knnRec(root, q, k, 0, distFn, heap);
            List<Pair> out = new ArrayList<>(heap);
            out.sort(Comparator.comparingDouble(a -> a.dist));
            return out;
        }

        private void knnRec(Node node, List<Float> q, int k, int depth, DistFn distFn, PriorityQueue<Pair> heap) {
            if (node == null) {
                return;
            }
            float dn = distFn.apply(q, node.item.emb);
            if (heap.size() < k || dn < heap.peek().dist) {
                heap.offer(new Pair(dn, node.item.id));
                if (heap.size() > k) {
                    heap.poll();
                }
            }
            int axis = depth % dims;
            float diff = q.get(axis) - node.item.emb.get(axis);
            Node closer = diff < 0 ? node.left : node.right;
            Node farther = diff < 0 ? node.right : node.left;
            knnRec(closer, q, k, depth + 1, distFn, heap);
            if (heap.size() < k || Math.abs(diff) < heap.peek().dist) {
                knnRec(farther, q, k, depth + 1, distFn, heap);
            }
        }
    }

    static final class HNSW {
        static final class Node {
            final VectorItem item;
            final int maxLayer;
            final List<List<Integer>> nbrs;

            Node(VectorItem item, int maxLayer) {
                this.item = item;
                this.maxLayer = maxLayer;
                this.nbrs = new ArrayList<>(maxLayer + 1);
                for (int i = 0; i <= maxLayer; i++) {
                    nbrs.add(new ArrayList<>());
                }
            }
        }

        static final class GraphInfo {
            int topLayer;
            int nodeCount;
            List<Integer> nodesPerLayer = new ArrayList<>();
            List<Integer> edgesPerLayer = new ArrayList<>();
            List<Map<String, Object>> nodes = new ArrayList<>();
            List<Map<String, Object>> edges = new ArrayList<>();
        }

        private final Map<Integer, Node> graph = new HashMap<>();
        private final int m;
        private final int m0;
        private final int efBuild;
        private final float ml;
        private final Random random = new Random(42);
        private int topLayer = -1;
        private int entryPoint = -1;

        HNSW(int m, int efBuild) {
            this.m = m;
            this.m0 = 2 * m;
            this.efBuild = efBuild;
            this.ml = (float) (1.0 / Math.log(m));
        }

        private int randLevel() {
            double u = Math.max(1e-8, random.nextDouble());
            return (int) Math.floor(-Math.log(u) * ml);
        }

        void insert(VectorItem item, DistFn distFn) {
            int id = item.id;
            int lvl = randLevel();
            graph.put(id, new Node(item, lvl));

            if (entryPoint == -1) {
                entryPoint = id;
                topLayer = lvl;
                return;
            }

            int ep = entryPoint;
            for (int lc = topLayer; lc > lvl; lc--) {
                if (lc < graph.get(ep).nbrs.size()) {
                    List<Pair> w = searchLayer(item.emb, ep, 1, lc, distFn);
                    if (!w.isEmpty()) {
                        ep = w.get(0).id;
                    }
                }
            }

            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<Pair> w = searchLayer(item.emb, ep, efBuild, lc, distFn);
                int maxM = lc == 0 ? m0 : m;
                List<Integer> sel = selectNeighbors(w, maxM);
                graph.get(id).nbrs.set(lc, new ArrayList<>(sel));

                for (int nid : sel) {
                    Node neighbor = graph.get(nid);
                    if (neighbor == null) {
                        continue;
                    }
                    while (neighbor.nbrs.size() <= lc) {
                        neighbor.nbrs.add(new ArrayList<>());
                    }
                    List<Integer> conn = neighbor.nbrs.get(lc);
                    conn.add(id);
                    if (conn.size() > maxM) {
                        List<Pair> ds = new ArrayList<>();
                        for (int c : conn) {
                            Node cn = graph.get(c);
                            if (cn != null) {
                                ds.add(new Pair(distFn.apply(neighbor.item.emb, cn.item.emb), c));
                            }
                        }
                        ds.sort(Comparator.comparingDouble(a -> a.dist));
                        conn.clear();
                        for (int i = 0; i < Math.min(maxM, ds.size()); i++) {
                            conn.add(ds.get(i).id);
                        }
                    }
                }
                if (!w.isEmpty()) {
                    ep = w.get(0).id;
                }
            }

            if (lvl > topLayer) {
                topLayer = lvl;
                entryPoint = id;
            }
        }

        List<Pair> knn(List<Float> q, int k, int ef, DistFn distFn) {
            if (entryPoint == -1) {
                return Collections.emptyList();
            }
            int ep = entryPoint;
            for (int lc = topLayer; lc > 0; lc--) {
                Node cur = graph.get(ep);
                if (cur != null && lc < cur.nbrs.size()) {
                    List<Pair> w = searchLayer(q, ep, 1, lc, distFn);
                    if (!w.isEmpty()) {
                        ep = w.get(0).id;
                    }
                }
            }
            List<Pair> w = searchLayer(q, ep, Math.max(ef, k), 0, distFn);
            if (w.size() > k) {
                return new ArrayList<>(w.subList(0, k));
            }
            return w;
        }

        void remove(int id) {
            if (!graph.containsKey(id)) {
                return;
            }
            for (Node node : graph.values()) {
                for (List<Integer> layer : node.nbrs) {
                    layer.removeIf(v -> v == id);
                }
            }
            if (entryPoint == id) {
                entryPoint = -1;
                for (int nid : graph.keySet()) {
                    if (nid != id) {
                        entryPoint = nid;
                        break;
                    }
                }
            }
            graph.remove(id);
        }

        GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer = topLayer;
            gi.nodeCount = graph.size();
            int maxL = Math.max(topLayer + 1, 1);
            for (int i = 0; i < maxL; i++) {
                gi.nodesPerLayer.add(0);
                gi.edgesPerLayer.add(0);
            }
            for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
                int id = entry.getKey();
                Node node = entry.getValue();
                Map<String, Object> n = new LinkedHashMap<>();
                n.put("id", id);
                n.put("metadata", node.item.metadata);
                n.put("category", node.item.category);
                n.put("maxLyr", node.maxLayer);
                gi.nodes.add(n);

                for (int lc = 0; lc <= node.maxLayer && lc < maxL; lc++) {
                    gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                    if (lc < node.nbrs.size()) {
                        for (int nid : node.nbrs.get(lc)) {
                            if (id < nid) {
                                gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                                Map<String, Object> e = new LinkedHashMap<>();
                                e.put("src", id);
                                e.put("dst", nid);
                                e.put("lyr", lc);
                                gi.edges.add(e);
                            }
                        }
                    }
                }
            }
            return gi;
        }

        int size() {
            return graph.size();
        }

        private List<Pair> searchLayer(List<Float> q, int ep, int ef, int layer, DistFn distFn) {
            Set<Integer> visited = new HashSet<>();
            PriorityQueue<Pair> candidates = new PriorityQueue<>(Comparator.comparingDouble(a -> a.dist));
            PriorityQueue<Pair> found = new PriorityQueue<>(Comparator.comparingDouble(a -> -a.dist));

            Node epNode = graph.get(ep);
            if (epNode == null) {
                return Collections.emptyList();
            }
            float d0 = distFn.apply(q, epNode.item.emb);
            visited.add(ep);
            candidates.offer(new Pair(d0, ep));
            found.offer(new Pair(d0, ep));

            while (!candidates.isEmpty()) {
                Pair cur = candidates.poll();
                if (found.size() >= ef && cur.dist > found.peek().dist) {
                    break;
                }
                Node curNode = graph.get(cur.id);
                if (curNode == null || layer >= curNode.nbrs.size()) {
                    continue;
                }
                for (int nid : curNode.nbrs.get(layer)) {
                    if (visited.contains(nid) || !graph.containsKey(nid)) {
                        continue;
                    }
                    visited.add(nid);
                    float nd = distFn.apply(q, graph.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().dist) {
                        candidates.offer(new Pair(nd, nid));
                        found.offer(new Pair(nd, nid));
                        if (found.size() > ef) {
                            found.poll();
                        }
                    }
                }
            }

            List<Pair> res = new ArrayList<>(found);
            res.sort(Comparator.comparingDouble(a -> a.dist));
            return res;
        }

        private List<Integer> selectNeighbors(List<Pair> cands, int maxM) {
            List<Integer> out = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++) {
                out.add(cands.get(i).id);
            }
            return out;
        }
    }

    static final class VectorDB {
        static final class Hit {
            int id;
            String meta;
            String cat;
            List<Float> emb;
            float dist;
        }

        static final class SearchOut {
            List<Hit> hits = new ArrayList<>();
            long latencyUs;
            String algo;
            String metric;
        }

        static final class BenchOut {
            long bfUs;
            long kdUs;
            long hnswUs;
            int n;
        }

        private final Map<Integer, VectorItem> store = new HashMap<>();
        private final BruteForce bf = new BruteForce();
        private final KDTree kdTree;
        private final HNSW hnsw = new HNSW(16, 200);
        private int nextId = 1;

        VectorDB(int dims) {
            this.kdTree = new KDTree(dims);
        }

        synchronized int insert(String meta, String cat, List<Float> emb, DistFn distFn) {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdTree.insert(v);
            hnsw.insert(v, distFn);
            return v.id;
        }

        synchronized boolean remove(int id) {
            if (!store.containsKey(id)) {
                return false;
            }
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdTree.rebuild(new ArrayList<>(store.values()));
            return true;
        }

        synchronized SearchOut search(List<Float> q, int k, String metric, String algo) {
            DistFn distFn = Distance.get(metric);
            long start = System.nanoTime();
            List<Pair> raw;
            if ("bruteforce".equals(algo)) {
                raw = bf.knn(q, k, distFn);
            } else if ("kdtree".equals(algo)) {
                raw = kdTree.knn(q, k, distFn);
            } else {
                raw = hnsw.knn(q, k, 50, distFn);
            }
            long us = (System.nanoTime() - start) / 1000L;

            SearchOut out = new SearchOut();
            out.latencyUs = us;
            out.algo = algo;
            out.metric = metric;
            for (Pair pair : raw) {
                VectorItem item = store.get(pair.id);
                if (item != null) {
                    Hit hit = new Hit();
                    hit.id = item.id;
                    hit.meta = item.metadata;
                    hit.cat = item.category;
                    hit.emb = item.emb;
                    hit.dist = pair.dist;
                    out.hits.add(hit);
                }
            }
            return out;
        }

        synchronized BenchOut benchmark(List<Float> q, int k, String metric) {
            DistFn distFn = Distance.get(metric);
            BenchOut out = new BenchOut();
            out.bfUs = timeUs(() -> bf.knn(q, k, distFn));
            out.kdUs = timeUs(() -> kdTree.knn(q, k, distFn));
            out.hnswUs = timeUs(() -> hnsw.knn(q, k, 50, distFn));
            out.n = store.size();
            return out;
        }

        synchronized List<VectorItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized HNSW.GraphInfo hnswInfo() {
            return hnsw.getInfo();
        }

        synchronized int size() {
            return store.size();
        }
    }

    static final class DocumentDB {
        private final Map<Integer, DocItem> store = new HashMap<>();
        private final HNSW hnsw = new HNSW(16, 200);
        private final BruteForce bf = new BruteForce();
        private int nextId = 1;
        private int dims = 0;

        synchronized int insert(String title, String text, List<Float> emb) {
            if (dims == 0) {
                dims = emb.size();
            }
            DocItem item = new DocItem(nextId++, title, text, emb);
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", emb);
            hnsw.insert(vi, Distance::cosine);
            bf.insert(vi);
            return item.id;
        }

        synchronized List<Map<String, Object>> search(List<Float> q, int k, float maxDist) {
            if (store.isEmpty()) {
                return Collections.emptyList();
            }
            List<Pair> raw = store.size() < 10 ? bf.knn(q, k, Distance::cosine) : hnsw.knn(q, k, 50, Distance::cosine);
            List<Map<String, Object>> out = new ArrayList<>();
            for (Pair pair : raw) {
                DocItem item = store.get(pair.id);
                if (item != null && pair.dist <= maxDist) {
                    Map<String, Object> row = new LinkedHashMap<>();
                    row.put("distance", pair.dist);
                    row.put("item", item);
                    out.add(row);
                }
            }
            return out;
        }

        synchronized boolean remove(int id) {
            if (!store.containsKey(id)) {
                return false;
            }
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        }

        synchronized List<DocItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized int size() {
            return store.size();
        }

        synchronized int getDims() {
            return dims;
        }
    }

    static final class JsonUtil {
        static String escape(String s) {
            StringBuilder sb = new StringBuilder("\"");
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default -> sb.append(c);
                }
            }
            return sb.append('"').toString();
        }

        static String vec(List<Float> v) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < v.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(String.format(Locale.US, "%.4f", v.get(i)));
            }
            return sb.append(']').toString();
        }

        static String extractStr(String body, String key) {
            int p = body.indexOf("\"" + key + "\"");
            if (p < 0) {
                return "";
            }
            p = body.indexOf(':', p);
            if (p < 0) {
                return "";
            }
            p++;
            while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) {
                p++;
            }
            if (p >= body.length() || body.charAt(p) != '"') {
                return "";
            }
            p++;
            StringBuilder out = new StringBuilder();
            while (p < body.length()) {
                char c = body.charAt(p);
                if (c == '"') {
                    break;
                }
                if (c == '\\' && p + 1 < body.length()) {
                    p++;
                    char e = body.charAt(p);
                    switch (e) {
                        case '"' -> out.append('"');
                        case '\\' -> out.append('\\');
                        case 'n' -> out.append('\n');
                        case 'r' -> out.append('\r');
                        case 't' -> out.append('\t');
                        default -> out.append(e);
                    }
                } else {
                    out.append(c);
                }
                p++;
            }
            return out.toString();
        }

        static int extractInt(String body, String key, int def) {
            int p = body.indexOf("\"" + key + "\"");
            if (p < 0) {
                return def;
            }
            p = body.indexOf(':', p);
            if (p < 0) {
                return def;
            }
            p++;
            while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) {
                p++;
            }
            int end = p;
            while (end < body.length() && (Character.isDigit(body.charAt(end)) || body.charAt(end) == '-')) {
                end++;
            }
            try {
                return Integer.parseInt(body.substring(p, end));
            } catch (Exception e) {
                return def;
            }
        }

        static List<Float> extractFloatArray(String body, String key) {
            int p = body.indexOf("\"" + key + "\"");
            if (p < 0) {
                return Collections.emptyList();
            }
            p = body.indexOf('[', p);
            if (p < 0) {
                return Collections.emptyList();
            }
            int e = body.indexOf(']', p);
            if (e < 0) {
                return Collections.emptyList();
            }
            return parseVec(body.substring(p + 1, e));
        }

        static List<Float> parseVec(String s) {
            String[] parts = s.split(",");
            List<Float> out = new ArrayList<>();
            for (String part : parts) {
                String t = part.trim();
                if (!t.isEmpty()) {
                    try {
                        out.add(Float.parseFloat(t));
                    } catch (Exception ignored) {
                    }
                }
            }
            return out;
        }
    }

    static final class OllamaClient {
        private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
        private final String baseUrl;
        String embedModel = "nomic-embed-text";
        String genModel = "llama3.2";

        OllamaClient(String host, int port) {
            this.baseUrl = "http://" + host + ":" + port;
        }

        boolean isAvailable() {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/api/tags"))
                        .timeout(Duration.ofSeconds(3))
                        .GET()
                        .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                return res.statusCode() == 200;
            } catch (Exception e) {
                return false;
            }
        }

        List<Float> embed(String text) {
            String body = "{\"model\":\"" + esc(embedModel, true) + "\",\"prompt\":\"" + esc(text, false) + "\"}";
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/api/embeddings"))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(40))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) {
                    return Collections.emptyList();
                }
                return parseEmbedding(res.body());
            } catch (Exception e) {
                return Collections.emptyList();
            }
        }

        String generate(String prompt) {
            String body = "{\"model\":\"" + esc(genModel, true) + "\",\"prompt\":\"" + esc(prompt, false) + "\",\"stream\":false}";
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/api/generate"))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(240))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build();
                HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
                if (res.statusCode() != 200) {
                    return "ERROR: Ollama unavailable. Run: ollama serve";
                }
                return JsonUtil.extractStr(res.body(), "response");
            } catch (Exception e) {
                return "ERROR: Ollama unavailable. Run: ollama serve";
            }
        }

        private List<Float> parseEmbedding(String body) {
            int p = body.indexOf("\"embedding\"");
            if (p < 0) {
                return Collections.emptyList();
            }
            p = body.indexOf('[', p);
            if (p < 0) {
                return Collections.emptyList();
            }
            int e = p + 1;
            int depth = 1;
            while (e < body.length() && depth > 0) {
                char c = body.charAt(e);
                if (c == '[') {
                    depth++;
                } else if (c == ']') {
                    depth--;
                }
                e++;
            }
            if (e <= p + 1) {
                return Collections.emptyList();
            }
            return JsonUtil.parseVec(body.substring(p + 1, e - 1));
        }

        private String esc(String s, boolean model) {
            if (model) {
                return s.replace("\\", "\\\\").replace("\"", "\\\"");
            }
            return s.replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t");
        }
    }

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] ws = text.trim().split("\\s+");
        if (ws.length == 0 || (ws.length == 1 && ws[0].isEmpty())) {
            return Collections.emptyList();
        }
        if (ws.length <= chunkWords) {
            return Collections.singletonList(text);
        }
        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < ws.length; i += step) {
            int end = Math.min(i + chunkWords, ws.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) {
                    sb.append(' ');
                }
                sb.append(ws[j]);
            }
            chunks.add(sb.toString());
            if (end == ws.length) {
                break;
            }
        }
        return chunks;
    }

    static long timeUs(Runnable r) {
        long t0 = System.nanoTime();
        r.run();
        return (System.nanoTime() - t0) / 1000L;
    }

    static String readBody(InputStream body) throws IOException {
        return new String(body.readAllBytes(), StandardCharsets.UTF_8);
    }

    static void addCors(Headers headers) {
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        headers.set("Access-Control-Allow-Headers", "Content-Type");
    }

    static void sendJson(HttpExchange ex, int status, String json) throws IOException {
        Headers h = ex.getResponseHeaders();
        addCors(h);
        h.set("Content-Type", "application/json; charset=utf-8");
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    static void sendText(HttpExchange ex, int status, String body, String contentType) throws IOException {
        Headers h = ex.getResponseHeaders();
        addCors(h);
        h.set("Content-Type", contentType);
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    static Map<String, String> parseQuery(URI uri) {
        Map<String, String> out = new HashMap<>();
        String q = uri.getRawQuery();
        if (q == null || q.isEmpty()) {
            return out;
        }
        String[] parts = q.split("&");
        for (String part : parts) {
            String[] kv = part.split("=", 2);
            String key = URLDecoder.decode(kv[0], StandardCharsets.UTF_8);
            String val = kv.length > 1 ? URLDecoder.decode(kv[1], StandardCharsets.UTF_8) : "";
            out.put(key, val);
        }
        return out;
    }

    static void loadDemo(VectorDB db) {
        DistFn dist = Distance.get("cosine");
        db.insert("Linked List: nodes connected by pointers", "cs", List.of(0.90f, 0.85f, 0.72f, 0.68f, 0.12f, 0.08f, 0.15f, 0.10f, 0.05f, 0.08f, 0.06f, 0.09f, 0.07f, 0.11f, 0.08f, 0.06f), dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs", List.of(0.88f, 0.82f, 0.78f, 0.74f, 0.15f, 0.10f, 0.08f, 0.12f, 0.06f, 0.07f, 0.08f, 0.05f, 0.09f, 0.06f, 0.07f, 0.10f), dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs", List.of(0.82f, 0.76f, 0.88f, 0.80f, 0.20f, 0.18f, 0.12f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.08f, 0.09f, 0.06f, 0.07f), dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs", List.of(0.85f, 0.80f, 0.75f, 0.82f, 0.18f, 0.14f, 0.10f, 0.08f, 0.06f, 0.09f, 0.07f, 0.06f, 0.10f, 0.08f, 0.09f, 0.07f), dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs", List.of(0.87f, 0.78f, 0.70f, 0.76f, 0.13f, 0.11f, 0.09f, 0.14f, 0.08f, 0.07f, 0.06f, 0.08f, 0.07f, 0.10f, 0.08f, 0.09f), dist);
        db.insert("Calculus: derivatives integrals and limits", "math", List.of(0.12f, 0.15f, 0.18f, 0.10f, 0.91f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.07f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f), dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math", List.of(0.20f, 0.18f, 0.15f, 0.12f, 0.88f, 0.90f, 0.82f, 0.76f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f, 0.07f, 0.08f, 0.09f), dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math", List.of(0.15f, 0.12f, 0.20f, 0.18f, 0.84f, 0.80f, 0.88f, 0.82f, 0.07f, 0.08f, 0.06f, 0.10f, 0.09f, 0.06f, 0.09f, 0.08f), dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math", List.of(0.22f, 0.16f, 0.14f, 0.20f, 0.80f, 0.85f, 0.76f, 0.90f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.10f, 0.07f, 0.06f), dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math", List.of(0.18f, 0.20f, 0.16f, 0.14f, 0.86f, 0.78f, 0.84f, 0.80f, 0.06f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.10f, 0.07f), dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", List.of(0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.90f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.09f, 0.07f), dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food", List.of(0.06f, 0.08f, 0.07f, 0.09f, 0.09f, 0.06f, 0.08f, 0.07f, 0.86f, 0.90f, 0.82f, 0.76f, 0.07f, 0.09f, 0.06f, 0.08f), dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", List.of(0.09f, 0.07f, 0.06f, 0.08f, 0.08f, 0.09f, 0.07f, 0.06f, 0.82f, 0.78f, 0.90f, 0.84f, 0.09f, 0.07f, 0.08f, 0.06f), dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food", List.of(0.07f, 0.09f, 0.08f, 0.06f, 0.06f, 0.07f, 0.09f, 0.08f, 0.78f, 0.82f, 0.86f, 0.90f, 0.06f, 0.08f, 0.07f, 0.09f), dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food", List.of(0.06f, 0.07f, 0.10f, 0.09f, 0.10f, 0.06f, 0.07f, 0.10f, 0.85f, 0.80f, 0.76f, 0.82f, 0.09f, 0.07f, 0.10f, 0.06f), dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", List.of(0.09f, 0.07f, 0.08f, 0.10f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.09f, 0.06f, 0.91f, 0.85f, 0.78f, 0.72f), dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports", List.of(0.07f, 0.09f, 0.06f, 0.08f, 0.09f, 0.07f, 0.10f, 0.08f, 0.07f, 0.09f, 0.08f, 0.07f, 0.87f, 0.89f, 0.82f, 0.76f), dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", List.of(0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.09f, 0.06f, 0.07f, 0.08f, 0.83f, 0.80f, 0.88f, 0.82f), dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports", List.of(0.25f, 0.20f, 0.22f, 0.18f, 0.22f, 0.18f, 0.20f, 0.15f, 0.06f, 0.08f, 0.07f, 0.09f, 0.80f, 0.84f, 0.78f, 0.90f), dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports", List.of(0.06f, 0.08f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.07f, 0.10f, 0.08f, 0.06f, 0.07f, 0.85f, 0.82f, 0.86f, 0.80f), dist);
    }

    static final class RouterHandler implements HttpHandler {
        private final VectorDB db;
        private final DocumentDB docDB;
        private final OllamaClient ollama;
        private final Pattern deletePattern = Pattern.compile("^/delete/(\\d+)$");
        private final Pattern docDeletePattern = Pattern.compile("^/doc/delete/(\\d+)$");

        RouterHandler(VectorDB db, DocumentDB docDB, OllamaClient ollama) {
            this.db = db;
            this.docDB = docDB;
            this.ollama = ollama;
        }

        @Override
        public void handle(HttpExchange ex) throws IOException {
            String method = ex.getRequestMethod();
            String path = ex.getRequestURI().getPath();

            if ("OPTIONS".equals(method)) {
                addCors(ex.getResponseHeaders());
                ex.sendResponseHeaders(204, -1);
                return;
            }

            if ("GET".equals(method) && "/".equals(path)) {
                Path file = Path.of("index.html");
                if (!Files.exists(file)) {
                    sendText(ex, 404, "index.html not found", "text/plain; charset=utf-8");
                    return;
                }
                sendText(ex, 200, Files.readString(file), "text/html; charset=utf-8");
                return;
            }

            try {
                if ("GET".equals(method) && "/search".equals(path)) {
                    handleSearch(ex);
                    return;
                }
                if ("POST".equals(method) && "/insert".equals(path)) {
                    handleInsert(ex);
                    return;
                }
                Matcher delMatch = deletePattern.matcher(path);
                if ("DELETE".equals(method) && delMatch.matches()) {
                    int id = Integer.parseInt(delMatch.group(1));
                    boolean ok = db.remove(id);
                    sendJson(ex, 200, "{\"ok\":" + ok + "}");
                    return;
                }
                if ("GET".equals(method) && "/items".equals(path)) {
                    handleItems(ex);
                    return;
                }
                if ("GET".equals(method) && "/benchmark".equals(path)) {
                    handleBenchmark(ex);
                    return;
                }
                if ("GET".equals(method) && "/hnsw-info".equals(path)) {
                    handleHnswInfo(ex);
                    return;
                }
                if ("POST".equals(method) && "/doc/insert".equals(path)) {
                    handleDocInsert(ex);
                    return;
                }
                Matcher docDel = docDeletePattern.matcher(path);
                if ("DELETE".equals(method) && docDel.matches()) {
                    int id = Integer.parseInt(docDel.group(1));
                    boolean ok = docDB.remove(id);
                    sendJson(ex, 200, "{\"ok\":" + ok + "}");
                    return;
                }
                if ("GET".equals(method) && "/doc/list".equals(path)) {
                    handleDocList(ex);
                    return;
                }
                if ("POST".equals(method) && "/doc/search".equals(path)) {
                    handleDocSearch(ex);
                    return;
                }
                if ("POST".equals(method) && "/doc/ask".equals(path)) {
                    handleDocAsk(ex);
                    return;
                }
                if ("GET".equals(method) && "/status".equals(path)) {
                    handleStatus(ex);
                    return;
                }
                if ("GET".equals(method) && "/stats".equals(path)) {
                    sendJson(ex, 200, "{\"count\":" + db.size() + ",\"dims\":" + DIMS
                            + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                            + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
                    return;
                }
                sendJson(ex, 404, "{\"error\":\"not found\"}");
            } catch (Exception e) {
                sendJson(ex, 500, "{\"error\":" + JsonUtil.escape("server error: " + e.getMessage()) + "}");
            }
        }

        private void handleSearch(HttpExchange ex) throws IOException {
            Map<String, String> q = parseQuery(ex.getRequestURI());
            List<Float> vec = JsonUtil.parseVec(q.getOrDefault("v", ""));
            if (vec.size() != DIMS) {
                sendJson(ex, 400, "{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k;
            try {
                k = Integer.parseInt(q.getOrDefault("k", "5"));
            } catch (Exception ignored) {
                k = 5;
            }
            String metric = q.getOrDefault("metric", "cosine");
            String algo = q.getOrDefault("algo", "hnsw");
            VectorDB.SearchOut out = db.search(vec, k, metric, algo);
            StringBuilder sb = new StringBuilder();
            sb.append("{\"results\":[");
            for (int i = 0; i < out.hits.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                VectorDB.Hit h = out.hits.get(i);
                sb.append("{\"id\":").append(h.id)
                        .append(",\"metadata\":").append(JsonUtil.escape(h.meta))
                        .append(",\"category\":").append(JsonUtil.escape(h.cat))
                        .append(",\"distance\":").append(String.format(Locale.US, "%.6f", h.dist))
                        .append(",\"embedding\":").append(JsonUtil.vec(h.emb)).append('}');
            }
            sb.append("],\"latencyUs\":").append(out.latencyUs)
                    .append(",\"algo\":").append(JsonUtil.escape(out.algo))
                    .append(",\"metric\":").append(JsonUtil.escape(out.metric))
                    .append('}');
            sendJson(ex, 200, sb.toString());
        }

        private void handleInsert(HttpExchange ex) throws IOException {
            String body = readBody(ex.getRequestBody());
            String meta = JsonUtil.extractStr(body, "metadata");
            String cat = JsonUtil.extractStr(body, "category");
            List<Float> emb = JsonUtil.extractFloatArray(body, "embedding");
            if (meta.isEmpty() || emb.size() != DIMS) {
                sendJson(ex, 400, "{\"error\":\"invalid body\"}");
                return;
            }
            int id = db.insert(meta, cat, emb, Distance.get("cosine"));
            sendJson(ex, 200, "{\"id\":" + id + "}");
        }

        private void handleItems(HttpExchange ex) throws IOException {
            List<VectorItem> items = db.all();
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                VectorItem v = items.get(i);
                sb.append("{\"id\":").append(v.id)
                        .append(",\"metadata\":").append(JsonUtil.escape(v.metadata))
                        .append(",\"category\":").append(JsonUtil.escape(v.category))
                        .append(",\"embedding\":").append(JsonUtil.vec(v.emb))
                        .append('}');
            }
            sb.append(']');
            sendJson(ex, 200, sb.toString());
        }

        private void handleBenchmark(HttpExchange ex) throws IOException {
            Map<String, String> q = parseQuery(ex.getRequestURI());
            List<Float> vec = JsonUtil.parseVec(q.getOrDefault("v", ""));
            if (vec.size() != DIMS) {
                sendJson(ex, 400, "{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k;
            try {
                k = Integer.parseInt(q.getOrDefault("k", "5"));
            } catch (Exception ignored) {
                k = 5;
            }
            String metric = q.getOrDefault("metric", "cosine");
            VectorDB.BenchOut b = db.benchmark(vec, k, metric);
            sendJson(ex, 200, "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                    + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}");
        }

        private void handleHnswInfo(HttpExchange ex) throws IOException {
            HNSW.GraphInfo gi = db.hnswInfo();
            StringBuilder sb = new StringBuilder();
            sb.append("{\"topLayer\":").append(gi.topLayer)
                    .append(",\"nodeCount\":").append(gi.nodeCount)
                    .append(",\"nodesPerLayer\":").append(intArray(gi.nodesPerLayer))
                    .append(",\"edgesPerLayer\":").append(intArray(gi.edgesPerLayer))
                    .append(",\"nodes\":[");
            for (int i = 0; i < gi.nodes.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                Map<String, Object> n = gi.nodes.get(i);
                sb.append("{\"id\":").append(n.get("id"))
                        .append(",\"metadata\":").append(JsonUtil.escape(String.valueOf(n.get("metadata"))))
                        .append(",\"category\":").append(JsonUtil.escape(String.valueOf(n.get("category"))))
                        .append(",\"maxLyr\":").append(n.get("maxLyr"))
                        .append('}');
            }
            sb.append("],\"edges\":[");
            for (int i = 0; i < gi.edges.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                Map<String, Object> e = gi.edges.get(i);
                sb.append("{\"src\":").append(e.get("src"))
                        .append(",\"dst\":").append(e.get("dst"))
                        .append(",\"lyr\":").append(e.get("lyr"))
                        .append('}');
            }
            sb.append("]}");
            sendJson(ex, 200, sb.toString());
        }

        private void handleDocInsert(HttpExchange ex) throws IOException {
            String body = readBody(ex.getRequestBody());
            String title = JsonUtil.extractStr(body, "title");
            String text = JsonUtil.extractStr(body, "text");
            if (title.isEmpty() || text.isEmpty()) {
                sendJson(ex, 400, "{\"error\":\"need title and text\"}");
                return;
            }
            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                List<Float> emb = ollama.embed(chunks.get(i));
                if (emb.isEmpty()) {
                    sendJson(ex, 500, "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                    return;
                }
                String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }
            StringBuilder sb = new StringBuilder("{\"ids\":[");
            for (int i = 0; i < ids.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(ids.get(i));
            }
            sb.append("],\"chunks\":").append(chunks.size())
                    .append(",\"dims\":").append(docDB.getDims()).append('}');
            sendJson(ex, 200, sb.toString());
        }

        private void handleDocList(HttpExchange ex) throws IOException {
            List<DocItem> docs = docDB.all();
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < docs.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                DocItem d = docs.get(i);
                String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
                int words = d.text.trim().isEmpty() ? 0 : d.text.trim().split("\\s+").length;
                sb.append("{\"id\":").append(d.id)
                        .append(",\"title\":").append(JsonUtil.escape(d.title))
                        .append(",\"preview\":").append(JsonUtil.escape(preview))
                        .append(",\"words\":").append(words)
                        .append('}');
            }
            sb.append(']');
            sendJson(ex, 200, sb.toString());
        }

        private void handleDocSearch(HttpExchange ex) throws IOException {
            String body = readBody(ex.getRequestBody());
            String question = JsonUtil.extractStr(body, "question");
            int k = JsonUtil.extractInt(body, "k", 3);
            if (question.isEmpty()) {
                sendJson(ex, 400, "{\"error\":\"need question\"}");
                return;
            }
            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                sendJson(ex, 500, "{\"error\":\"Ollama unavailable\"}");
                return;
            }
            List<Map<String, Object>> hits = docDB.search(qEmb, k, 0.7f);
            StringBuilder sb = new StringBuilder("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                float distance = (float) hits.get(i).get("distance");
                DocItem item = (DocItem) hits.get(i).get("item");
                sb.append("{\"id\":").append(item.id)
                        .append(",\"title\":").append(JsonUtil.escape(item.title))
                        .append(",\"distance\":").append(String.format(Locale.US, "%.4f", distance))
                        .append('}');
            }
            sb.append("]}");
            sendJson(ex, 200, sb.toString());
        }

        private void handleDocAsk(HttpExchange ex) throws IOException {
            String body = readBody(ex.getRequestBody());
            String question = JsonUtil.extractStr(body, "question");
            int k = JsonUtil.extractInt(body, "k", 3);
            if (question.isEmpty()) {
                sendJson(ex, 400, "{\"error\":\"need question\"}");
                return;
            }
            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                sendJson(ex, 500, "{\"error\":\"Ollama unavailable\"}");
                return;
            }
            List<Map<String, Object>> hits = docDB.search(qEmb, k, 0.7f);

            StringBuilder ctx = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                DocItem item = (DocItem) hits.get(i).get("item");
                ctx.append("[").append(i + 1).append("] ").append(item.title).append(":\n")
                        .append(item.text).append("\n\n");
            }

            String prompt = "You are a helpful assistant. Answer the user's question directly. "
                    + "Use the provided context if it contains relevant information. "
                    + "If it doesn't, just use your own general knowledge. "
                    + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                    + "Just answer the question naturally.\n\n"
                    + "Context:\n" + ctx
                    + "Question: " + question + "\n\n"
                    + "Answer:";

            String answer = ollama.generate(prompt);

            StringBuilder sb = new StringBuilder();
            sb.append("{\"answer\":").append(JsonUtil.escape(answer))
                    .append(",\"model\":").append(JsonUtil.escape(ollama.genModel))
                    .append(",\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                float distance = (float) hits.get(i).get("distance");
                DocItem item = (DocItem) hits.get(i).get("item");
                sb.append("{\"id\":").append(item.id)
                        .append(",\"title\":").append(JsonUtil.escape(item.title))
                        .append(",\"text\":").append(JsonUtil.escape(item.text))
                        .append(",\"distance\":").append(String.format(Locale.US, "%.4f", distance))
                        .append('}');
            }
            sb.append("],\"docCount\":").append(docDB.size()).append('}');
            sendJson(ex, 200, sb.toString());
        }

        private void handleStatus(HttpExchange ex) throws IOException {
            boolean up = ollama.isAvailable();
            String json = "{\"ollamaAvailable\":" + up
                    + ",\"embedModel\":" + JsonUtil.escape(ollama.embedModel)
                    + ",\"genModel\":" + JsonUtil.escape(ollama.genModel)
                    + ",\"docCount\":" + docDB.size()
                    + ",\"docDims\":" + docDB.getDims()
                    + ",\"demoDims\":" + DIMS
                    + ",\"demoCount\":" + db.size() + "}";
            sendJson(ex, 200, json);
        }

        private String intArray(List<Integer> values) {
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < values.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(values.get(i));
            }
            return sb.append(']').toString();
        }
    }

    public static void main(String[] args) throws IOException {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);
        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
        server.createContext("/", new RouterHandler(db, docDB, ollama));
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
    }
}
