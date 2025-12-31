#!/usr/bin/env python3
"""
Idle Agent - Never Waste GPU Cycles!

When pipeline is empty, do useful background work:
1. Prefetch knowledge - scrape/summarize trending topics
2. Build embeddings - pre-compute for fast retrieval
3. Pattern learning - analyze user behavior
4. Model warmup - keep GPU hot with micro-tasks
5. News digest - summarize latest from RSS feeds

All work is cached and ready when user asks related questions!
"""

import asyncio
import aiohttp
import feedparser
import time
import hashlib
import json
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta


@dataclass
class KnowledgeItem:
    """Cached knowledge from background work"""
    topic: str
    summary: str
    source: str
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


@dataclass
class IdleTask:
    """Background task for idle GPU"""
    task_type: str  # 'scrape', 'summarize', 'embed', 'warmup'
    priority: int   # 1=high, 5=low
    payload: dict
    created_at: float = field(default_factory=time.time)


class KnowledgeCache:
    """Cache of pre-computed knowledge"""

    def __init__(self, max_items: int = 1000):
        self.items: Dict[str, KnowledgeItem] = {}
        self.max_items = max_items

    def store(self, topic: str, summary: str, source: str, embedding: List[float] = None):
        key = hashlib.md5(topic.lower().encode()).hexdigest()[:16]
        self.items[key] = KnowledgeItem(
            topic=topic,
            summary=summary,
            source=source,
            embedding=embedding
        )
        self._prune()

    def lookup(self, query: str) -> Optional[KnowledgeItem]:
        """Find relevant cached knowledge"""
        key = hashlib.md5(query.lower().encode()).hexdigest()[:16]
        if key in self.items:
            self.items[key].access_count += 1
            return self.items[key]

        # Fuzzy match on topic keywords
        query_words = set(query.lower().split())
        for item in self.items.values():
            topic_words = set(item.topic.lower().split())
            overlap = len(query_words & topic_words)
            if overlap >= 2:  # At least 2 word match
                item.access_count += 1
                return item
        return None

    def _prune(self):
        """Remove old/unused items"""
        if len(self.items) <= self.max_items:
            return
        # Sort by access_count * recency
        scored = [
            (k, v.access_count / (v.age_hours + 1))
            for k, v in self.items.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = set(k for k, _ in scored[:self.max_items // 2])
        self.items = {k: v for k, v in self.items.items() if k in keep}


class RSSFetcher:
    """Fetch and parse RSS feeds for knowledge"""

    DEFAULT_FEEDS = [
        # Tech news
        ('https://news.ycombinator.com/rss', 'Hacker News'),
        ('https://feeds.arstechnica.com/arstechnica/technology-lab', 'Ars Technica'),
        ('https://www.reddit.com/r/MachineLearning/.rss', 'r/MachineLearning'),
        # Science
        ('https://www.nature.com/nature.rss', 'Nature'),
        # General
        ('https://feeds.bbci.co.uk/news/technology/rss.xml', 'BBC Tech'),
    ]

    def __init__(self, feeds: List[tuple] = None):
        self.feeds = feeds or self.DEFAULT_FEEDS
        self.last_fetch: Dict[str, float] = {}
        self.fetch_interval = 300  # 5 minutes minimum between fetches

    async def fetch_latest(self, max_items: int = 10) -> List[dict]:
        """Fetch latest items from all feeds"""
        items = []

        async with aiohttp.ClientSession() as session:
            for feed_url, source_name in self.feeds:
                # Rate limit
                if feed_url in self.last_fetch:
                    if time.time() - self.last_fetch[feed_url] < self.fetch_interval:
                        continue

                try:
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            feed = feedparser.parse(content)

                            for entry in feed.entries[:3]:  # Top 3 per feed
                                items.append({
                                    'title': entry.get('title', ''),
                                    'summary': entry.get('summary', '')[:500],
                                    'link': entry.get('link', ''),
                                    'source': source_name,
                                })

                            self.last_fetch[feed_url] = time.time()
                except Exception as e:
                    print(f"[IdleAgent] Feed error {source_name}: {e}")

        return items[:max_items]


class IdleAgent:
    """
    Background agent that keeps GPUs busy with useful work.

    When pipeline is idle:
    1. Check task queue for pending work
    2. Fetch latest news/knowledge
    3. Summarize with LLM
    4. Cache results for instant serving
    """

    def __init__(self, inference_fn, check_idle_fn):
        """
        inference_fn: async function to run inference (prompt -> response)
        check_idle_fn: function that returns True if pipeline is idle
        """
        self.inference_fn = inference_fn
        self.check_idle_fn = check_idle_fn

        self.knowledge_cache = KnowledgeCache()
        self.rss_fetcher = RSSFetcher()
        self.task_queue: Deque[IdleTask] = deque(maxlen=100)

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self.items_processed = 0
        self.summaries_generated = 0
        self.cache_hits = 0
        self.idle_time_used_ms = 0

    async def start(self):
        """Start idle agent loop"""
        self._running = True
        self._task = asyncio.create_task(self._idle_loop())
        print("[IdleAgent] Started - will use idle GPU time for knowledge building")

    async def stop(self):
        """Stop idle agent"""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _idle_loop(self):
        """Main loop - runs when GPU is idle"""
        while self._running:
            try:
                # Only work if pipeline is idle
                if not self.check_idle_fn():
                    await asyncio.sleep(0.5)
                    continue

                # Priority 1: Process queued tasks
                if self.task_queue:
                    task = self.task_queue.popleft()
                    await self._process_task(task)
                    continue

                # Priority 2: Fetch and summarize news
                await self._fetch_and_summarize()

                # Priority 3: Model warmup (keep GPU hot)
                await self._warmup()

                # Sleep before next cycle
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[IdleAgent] Error: {e}")
                await asyncio.sleep(10)

    async def _process_task(self, task: IdleTask):
        """Process a queued idle task"""
        start = time.perf_counter()

        if task.task_type == 'summarize':
            text = task.payload.get('text', '')
            topic = task.payload.get('topic', 'unknown')

            prompt = f"Summarize in 2-3 sentences: {text[:1000]}"
            summary = await self.inference_fn(prompt)

            self.knowledge_cache.store(topic, summary, 'user_request')
            self.summaries_generated += 1

        elif task.task_type == 'scrape':
            url = task.payload.get('url', '')
            # Would fetch and process URL
            pass

        self.idle_time_used_ms += (time.perf_counter() - start) * 1000
        self.items_processed += 1

    async def _fetch_and_summarize(self):
        """Fetch latest news and generate summaries"""
        items = await self.rss_fetcher.fetch_latest(5)

        for item in items:
            # Skip if already cached
            if self.knowledge_cache.lookup(item['title']):
                continue

            # Check if still idle
            if not self.check_idle_fn():
                return

            start = time.perf_counter()

            # Summarize with LLM
            prompt = f"Summarize this news in 1-2 sentences:\nTitle: {item['title']}\n{item['summary']}"

            try:
                summary = await self.inference_fn(prompt)

                self.knowledge_cache.store(
                    topic=item['title'],
                    summary=summary,
                    source=item['source']
                )

                self.summaries_generated += 1
                self.idle_time_used_ms += (time.perf_counter() - start) * 1000

                print(f"[IdleAgent] Cached: {item['title'][:50]}...")

            except Exception as e:
                print(f"[IdleAgent] Summary error: {e}")

    async def _warmup(self):
        """Keep model warm with micro-inference"""
        if not self.check_idle_fn():
            return

        # Quick inference to keep GPU active
        start = time.perf_counter()
        try:
            await self.inference_fn("Hello")  # Minimal warmup
            self.idle_time_used_ms += (time.perf_counter() - start) * 1000
        except:
            pass

    def queue_task(self, task_type: str, payload: dict, priority: int = 3):
        """Add task to idle queue"""
        task = IdleTask(task_type=task_type, priority=priority, payload=payload)
        self.task_queue.append(task)

    def check_knowledge(self, query: str) -> Optional[str]:
        """Check if we have cached knowledge for a query"""
        item = self.knowledge_cache.lookup(query)
        if item:
            self.cache_hits += 1
            return f"[Cached from {item.source}] {item.summary}"
        return None

    def get_stats(self) -> dict:
        return {
            'items_processed': self.items_processed,
            'summaries_generated': self.summaries_generated,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.knowledge_cache.items),
            'queue_depth': len(self.task_queue),
            'idle_time_used_ms': self.idle_time_used_ms,
        }


# ============================================================================
# Integration with Pipeline Server
# ============================================================================

def create_idle_agent_for_pipeline(pipeline_server):
    """
    Create an IdleAgent integrated with a PipelineServer.

    Usage:
        server = PipelineServer(...)
        agent = create_idle_agent_for_pipeline(server)
        await agent.start()
    """

    async def inference_fn(prompt: str) -> str:
        """Run inference through the pipeline"""
        # Create a simple inference request
        result = await pipeline_server.forward_prompt(prompt, max_tokens=50)
        return result.get('text', '')

    def check_idle_fn() -> bool:
        """Check if pipeline is idle"""
        stats = pipeline_server.get_stats()
        return stats.get('queue_depth', 0) == 0

    return IdleAgent(inference_fn, check_idle_fn)


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == '__main__':
    async def mock_inference(prompt):
        await asyncio.sleep(0.1)  # Simulate inference
        return f"Summary of: {prompt[:50]}..."

    def mock_idle():
        return True  # Always idle for testing

    async def test():
        agent = IdleAgent(mock_inference, mock_idle)
        await agent.start()

        # Let it run for a bit
        await asyncio.sleep(30)

        print("\n[IdleAgent] Stats:", agent.get_stats())

        # Test knowledge lookup
        result = agent.check_knowledge("machine learning")
        print(f"[IdleAgent] Knowledge lookup: {result}")

        await agent.stop()

    asyncio.run(test())
