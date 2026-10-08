"""Synthetic 500,000 document corpus generator matching PRD Sections 42 and 43."""

from __future__ import annotations

import argparse
import datetime
import json
import random
import time
from pathlib import Path

DOMAINS = {
    "cybersecurity": [
        "Zero Trust Network Access (ZTNA) enforces strict identity verification for every user and device attempting to access network resources, regardless of whether they are located inside or outside the enterprise network perimeter.",
        "Mutual TLS (mTLS) authentication provides bidirectional cryptographic proof of identity between client and server, preventing man-in-the-middle attacks across distributed microservices.",
        "Role-Based Access Control (RBAC) restricts system authorization to authorized users based on defined organisational roles, minimizing the blast radius of compromised credentials.",
        "Cryptographic key rotation policies mitigate the exposure window of compromised symmetric keys and digital certificates across cloud environments.",
        "Denial of Service (DoS) mitigation leverages distributed scrubbing centers, anycast routing, and rate-limiting token buckets to absorb volumetric network floods.",
    ],
    "networking": [
        "Border Gateway Protocol (BGP) manages packet routing across autonomous systems on the global Internet using path-vector algorithms and peering policies.",
        "TCP congestion control algorithms like BBR and CUBIC dynamically adjust congestion window sizes based on round-trip time and packet loss observations.",
        "Domain Name System (DNS) translates human-readable hostnames into 32-bit IPv4 or 128-bit IPv6 addresses via hierarchical recursive resolvers and authoritative nameservers.",
        "Virtual Extensible LAN (VXLAN) encapsulates Layer 2 Ethernet frames within Layer 4 UDP packets to scale multi-tenant overlay networks in modern data centers.",
        "Quality of Service (QoS) mechanisms prioritize latency-sensitive voice and video traffic over bulk file transfers using differentiated services code points (DSCP).",
    ],
    "cloud": [
        "Kubernetes Horizontal Pod Autoscaling (HPA) continuously monitors CPU, memory, and custom Prometheus metrics to dynamically adjust deployment replica counts.",
        "Infrastructure as Code (IaC) enables declarative specification of cloud infrastructure using automated version-controlled configuration templates.",
        "Serverless functions scale on-demand from zero to thousands of concurrent instances, executing isolated workloads with fine-grained consumption pricing.",
        "Multi-cloud architectures distribute critical services across independent cloud providers to achieve geographic redundancy and avoid vendor lock-in.",
        "Cloud object stores like Amazon S3 and Google Cloud Storage provide high durability through Reed-Solomon erasure coding and cross-region replication.",
    ],
    "ai": [
        "Dense vector embeddings represent textual tokens and semantic concepts as continuous numerical coordinates in high-dimensional geometric vector spaces.",
        "Hierarchical Navigable Small World (HNSW) graphs organize vectors into multi-layer proximity networks for approximate nearest neighbor search with logarithmic scaling.",
        "Retrieval-Augmented Generation (RAG) grounds language model generations in verified authoritative documents to eliminate hallucinations and enforce source attribution.",
        "Cross-encoder neural rerankers evaluate query and document pairs simultaneously with cross-attention to produce highly calibrated relevance rankings.",
        "Quantization techniques such as INT8 and FP8 reduce model memory footprints and accelerate vector dot-product computations on modern CPU SIMD extensions.",
    ],
    "software engineering": [
        "Event-driven architectures decouple distributed producers and consumers through asynchronous message brokers like Apache Kafka and RabbitMQ.",
        "Continuous Integration and Continuous Deployment (CI/CD) pipelines automate testing, linting, type-checking, and container deployment on every git commit.",
        "Database connection pooling maintains a cache of active database connections to avoid the high latency overhead of TCP handshakes on high-frequency transactions.",
        "Log-Structured Merge (LSM) trees optimize write performance by appending updates to immutable memtables and flushing sequentially to sorted string tables (SSTables).",
        "Clean architecture principles enforce separation of concerns, ensuring domain business logic remains decoupled from frameworks and database implementations.",
    ],
    "finance": [
        "High-frequency algorithmic trading systems utilize FPGA accelerators and direct market data feeds to execute trades with sub-microsecond latencies.",
        "Automated Clearing House (ACH) and real-time payment networks facilitate electronic credit and debit transfers between commercial banking institutions.",
        "Financial risk management models compute Value at Risk (VaR) and Expected Shortfall using historical simulations and Monte Carlo methods.",
        "Distributed ledger technologies provide tamper-evident cryptographic consensus for digital asset settlement and smart contract execution.",
        "Anti-Money Laundering (AML) transaction monitoring systems analyze graph connectivity patterns to identify suspicious structuring and fund movements.",
    ],
    "science": [
        "Cryo-electron microscopy resolves three-dimensional structures of macromolecular protein complexes at near-atomic spatial resolution.",
        "CRISPR-Cas9 gene editing enables targeted genetic modifications by guiding endonuclease enzymes to specific nucleotide sequences.",
        "Quantum computing algorithms like Shor and Grover promise polynomial speedups for integer factorization and unstructured search problems.",
        "Gravitational wave interferometers like LIGO detect sub-atomic spacetime perturbations caused by colliding binary black holes.",
        "Neural machine translation models trained on bilingual corpora translate natural languages using encoder-decoder sequence architectures.",
    ],
    "documentation": [
        "API specification documents describe available endpoints, request schemas, status codes, and authentication requirements according to OpenAPI standards.",
        "Standard Operating Procedures (SOP) define incident response protocols, on-call escalation policies, and post-mortem analysis frameworks.",
        "Architecture Decision Records (ADR) capture important architectural choices, technical context, alternatives considered, and engineering trade-offs.",
        "Software release notes detail new feature additions, breaking API changes, deprecations, performance improvements, and security patches.",
        "Developer onboarding documentation guides engineers through repository setup, environment configuration, and local build verification.",
    ],
    "business": [
        "Key Performance Indicators (KPIs) track organizational progress toward strategic goals including customer acquisition cost and net revenue retention.",
        "Enterprise Resource Planning (ERP) software integrates core business operations including supply chain, procurement, inventory, and human resources.",
        "Product-led growth (PLG) strategies prioritize user self-serve onboarding, organic virality, and product utility to drive commercial adoption.",
        "Customer relationship management (CRM) platforms unify sales pipeline tracking, customer interactions, and automated marketing campaigns.",
        "SaaS business models rely on predictable recurring subscription revenue, high annual contract values, and negative revenue churn.",
    ],
    "technology": [
        "Modern central processing units feature multi-level cache hierarchies with L1, L2, and shared L3 caches to minimize main memory access latency.",
        "Solid-state drives (SSDs) utilize NVMe protocols over PCIe buses to deliver millions of input/output operations per second (IOPS).",
        "Operating system kernels provide virtual memory management, process scheduling, and hardware abstraction layers for user applications.",
        "Direct Memory Access (DMA) allows hardware subsystems to read and write system RAM independently of the central processor.",
        "Compiler optimization passes perform dead code elimination, loop unrolling, and function inlining to maximize machine code efficiency.",
    ],
}

AUTHORS = [
    "Alice Chen",
    "Bob Smith",
    "Carlos Rodriguez",
    "Diana Prince",
    "Elena Rostov",
    "Frank Zhang",
    "Grace Hopper",
    "Henry Ford",
]
TENANTS = ["default", "tenant_alpha", "tenant_beta", "tenant_gamma"]
LANGUAGES = ["en", "en", "en", "es", "de", "fr"]


def generate_document_corpus(
    total_documents: int = 500000,
    output_dir: Path | str = "./data/raw",
    batch_file_size: int = 25000,
) -> None:
    """Generate synthetic heterogeneous documents into sharded JSONL files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains_list = list(DOMAINS.keys())
    random.seed(42)

    total_files = (total_documents + batch_file_size - 1) // batch_file_size
    print(
        f"Generating {total_documents:,} documents across {total_files} partition files in '{out_dir}'..."
    )

    start_time = time.time()
    docs_generated = 0

    for file_idx in range(total_files):
        batch_size = min(batch_file_size, total_documents - docs_generated)
        file_path = out_dir / f"documents_part_{file_idx + 1:03d}.jsonl"

        with open(file_path, "w", encoding="utf-8") as f:
            for i in range(batch_size):
                doc_num = docs_generated + i + 1
                domain = random.choice(domains_list)
                snippets = DOMAINS[domain]

                # Compose realistic document content (1 to 3 domain paragraphs)
                num_paras = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
                selected_paras = random.sample(snippets, min(num_paras, len(snippets)))

                # Add title & header
                title = f"{domain.replace('_', ' ').title()} Technical Overview {doc_num}"
                full_text = f"# {title}\n\n" + "\n\n".join(selected_paras)

                # Intentional duplicate document generation (every 100th doc duplicates a previous doc)
                if doc_num > 100 and doc_num % 100 == 0:
                    title = f"{domain.replace('_', ' ').title()} Duplicate Reference"
                    full_text = f"# {title}\n\n" + snippets[0]

                author = random.choice(AUTHORS)
                tenant = random.choice(TENANTS)
                lang = random.choice(LANGUAGES)

                # Random date in recent past
                days_ago = random.randint(1, 365)
                created_date = (
                    datetime.date.today() - datetime.timedelta(days=days_ago)
                ).isoformat()

                doc_entry = {
                    "id": f"raw_doc_{doc_num:07d}",
                    "title": title,
                    "text": full_text,
                    "source": f"kb-{domain}",
                    "tenant_id": tenant,
                    "metadata": {
                        "category": domain,
                        "author": author,
                        "language": lang,
                        "created_at": created_date,
                        "tags": [domain, "production", f"v{random.randint(1, 3)}"],
                    },
                }

                f.write(json.dumps(doc_entry) + "\n")

        docs_generated += batch_size
        elapsed = time.time() - start_time
        rate = docs_generated / elapsed if elapsed > 0 else 0
        print(
            f"  Partition {file_idx + 1:03d}/{total_files}: {docs_generated:,} / {total_documents:,} docs ({rate:,.0f} docs/sec)"
        )

    total_time = time.time() - start_time
    print(
        f"Completed: generated {docs_generated:,} documents in {total_time:.1f}s ({docs_generated / total_time:,.0f} docs/sec)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 500,000 synthetic documents for benchmarking"
    )
    parser.add_argument(
        "--documents", type=int, default=500000, help="Total number of documents to generate"
    )
    parser.add_argument(
        "--output", type=str, default="./data/raw", help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=25000, help="Number of documents per partition file"
    )
    args = parser.parse_args()

    generate_document_corpus(
        total_documents=args.documents,
        output_dir=args.output,
        batch_file_size=args.batch_size,
    )
