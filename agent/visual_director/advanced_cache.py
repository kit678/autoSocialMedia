"""
Advanced Caching System with SQLite

This module implements a comprehensive caching system using SQLite
for query results, assets, and entity canonicalization.
"""

import os
import json
import sqlite3
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading

from .asset_types import Asset


class AdvancedCache:
    """
    SQLite-based caching system for visual director.
    
    Provides:
    - Query result caching
    - Asset metadata storage
    - Entity canonicalization
    - Performance metrics
    """
    
    def __init__(self, cache_dir: str, ttl_hours: int = 24):
        """
        Initialize advanced cache.
        
        Args:
            cache_dir: Directory for cache database and files
            ttl_hours: Time-to-live for cached entries in hours
        """
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.db_path = os.path.join(cache_dir, 'visual_cache.db')
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Initialize database
        self._init_database()
        
        # Performance metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'query_time': 0.0
        }
    
    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
        except Exception as e:
            self._local.conn.rollback()
            raise e
        else:
            self._local.conn.commit()
    
    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            # Query cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    adapter TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            
            # Asset cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_cache (
                    asset_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    local_path TEXT,
                    file_hash TEXT,
                    metadata TEXT NOT NULL,
                    source TEXT NOT NULL,
                    license TEXT,
                    width INTEGER,
                    height INTEGER,
                    duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Entity canonicalization table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS canonical_entities (
                    entity_name TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    canonical_asset_id TEXT,
                    metadata TEXT,
                    confidence REAL DEFAULT 0.8,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (canonical_asset_id) REFERENCES asset_cache(asset_id)
                )
            """)
            
            # Usage statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT NOT NULL,
                    segment_id TEXT,
                    video_project TEXT,
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    performance_score REAL,
                    FOREIGN KEY (asset_id) REFERENCES asset_cache(asset_id)
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_expires ON query_cache(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_url ON asset_cache(url)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_source ON asset_cache(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON canonical_entities(entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_asset ON usage_stats(asset_id)")
    
    def cache_query_results(
        self, 
        query: str, 
        adapter: str, 
        results: List[Asset]
    ) -> None:
        """
        Cache search query results.
        
        Args:
            query: The search query
            adapter: Name of the adapter that performed the search
            results: List of Asset objects returned
        """
        query_hash = self._hash_query(query, adapter)
        expires_at = datetime.now() + timedelta(hours=self.ttl_hours)
        
        # Serialize assets
        serialized_results = [self._serialize_asset(asset) for asset in results]
        results_json = json.dumps(serialized_results)
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO query_cache 
                (query_hash, query_text, adapter, results, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 
                    COALESCE((SELECT hit_count FROM query_cache WHERE query_hash = ?), 0))
            """, (query_hash, query, adapter, results_json, expires_at, query_hash))
    
    def get_cached_query_results(
        self, 
        query: str, 
        adapter: str
    ) -> Optional[List[Asset]]:
        """
        Retrieve cached query results if available.
        
        Args:
            query: The search query
            adapter: Name of the adapter
            
        Returns:
            List of Asset objects or None if not cached/expired
        """
        query_hash = self._hash_query(query, adapter)
        
        with self._get_connection() as conn:
            # Check for valid cached results
            row = conn.execute("""
                SELECT results FROM query_cache
                WHERE query_hash = ? AND expires_at > datetime('now')
            """, (query_hash,)).fetchone()
            
            if row:
                # Update hit count
                conn.execute("""
                    UPDATE query_cache 
                    SET hit_count = hit_count + 1 
                    WHERE query_hash = ?
                """, (query_hash,))
                
                self.metrics['hits'] += 1
                
                # Deserialize assets
                results_data = json.loads(row['results'])
                return [self._deserialize_asset(data) for data in results_data]
            
            self.metrics['misses'] += 1
            return None
    
    def cache_asset(self, asset: Asset) -> None:
        """
        Cache asset metadata and file information.
        
        Args:
            asset: Asset object to cache
        """
        metadata_json = json.dumps(asset.metadata)
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO asset_cache
                (asset_id, url, local_path, file_hash, metadata, source, 
                 license, width, height, duration, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'),
                    COALESCE((SELECT access_count + 1 FROM asset_cache WHERE asset_id = ?), 1))
            """, (
                asset.id, asset.url, asset.local_path,
                self._hash_file(asset.local_path) if asset.local_path else None,
                metadata_json, asset.source, asset.licence,
                asset.dimensions[0], asset.dimensions[1],
                asset.duration, asset.id
            ))
    
    def get_cached_asset(self, asset_id: str) -> Optional[Asset]:
        """
        Retrieve cached asset by ID.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Asset object or None if not cached
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM asset_cache WHERE asset_id = ?
            """, (asset_id,)).fetchone()
            
            if row:
                # Update access timestamp
                conn.execute("""
                    UPDATE asset_cache 
                    SET last_accessed = datetime('now'), 
                        access_count = access_count + 1
                    WHERE asset_id = ?
                """, (asset_id,))
                
                return self._row_to_asset(row)
            
            return None
    
    def set_canonical_entity(
        self, 
        entity_name: str, 
        entity_type: str,
        asset: Asset,
        confidence: float = 0.8
    ) -> None:
        """
        Set the canonical visual representation for an entity.
        
        Args:
            entity_name: Name of the entity (e.g., "OpenAI")
            entity_type: Type of entity (e.g., "company", "person")
            asset: The canonical asset for this entity
            confidence: Confidence score for this mapping
        """
        # Ensure asset is cached first
        self.cache_asset(asset)
        
        metadata = {
            'source': asset.source,
            'url': asset.url,
            'last_verified': datetime.now().isoformat()
        }
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO canonical_entities
                (entity_name, entity_type, canonical_asset_id, metadata, 
                 confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (
                entity_name.lower(), entity_type, asset.id,
                json.dumps(metadata), confidence
            ))
    
    def get_canonical_entity(
        self, 
        entity_name: str, 
        entity_type: Optional[str] = None
    ) -> Optional[Asset]:
        """
        Get the canonical visual representation for an entity.
        
        Args:
            entity_name: Name of the entity
            entity_type: Optional type filter
            
        Returns:
            Canonical Asset or None
        """
        with self._get_connection() as conn:
            if entity_type:
                query = """
                    SELECT a.* FROM canonical_entities e
                    JOIN asset_cache a ON e.canonical_asset_id = a.asset_id
                    WHERE e.entity_name = ? AND e.entity_type = ?
                """
                params = (entity_name.lower(), entity_type)
            else:
                query = """
                    SELECT a.* FROM canonical_entities e
                    JOIN asset_cache a ON e.canonical_asset_id = a.asset_id
                    WHERE e.entity_name = ?
                    ORDER BY e.confidence DESC
                    LIMIT 1
                """
                params = (entity_name.lower(),)
            
            row = conn.execute(query, params).fetchone()
            
            if row:
                return self._row_to_asset(row)
            
            return None
    
    def record_usage(
        self, 
        asset: Asset, 
        segment_id: str,
        video_project: str,
        performance_score: float = 0.0
    ) -> None:
        """
        Record asset usage for analytics.
        
        Args:
            asset: Asset that was used
            segment_id: ID of the segment where used
            video_project: Project/video identifier
            performance_score: How well the asset performed (0-1)
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO usage_stats
                (asset_id, segment_id, video_project, performance_score)
                VALUES (?, ?, ?, ?)
            """, (asset.id, segment_id, video_project, performance_score))
    
    def get_popular_assets(
        self, 
        source: Optional[str] = None,
        limit: int = 20
    ) -> List[Tuple[Asset, int]]:
        """
        Get most frequently used assets.
        
        Args:
            source: Optional filter by source
            limit: Maximum number of results
            
        Returns:
            List of (Asset, usage_count) tuples
        """
        with self._get_connection() as conn:
            if source:
                query = """
                    SELECT a.*, COUNT(u.id) as usage_count
                    FROM asset_cache a
                    LEFT JOIN usage_stats u ON a.asset_id = u.asset_id
                    WHERE a.source = ?
                    GROUP BY a.asset_id
                    ORDER BY usage_count DESC
                    LIMIT ?
                """
                params = (source, limit)
            else:
                query = """
                    SELECT a.*, COUNT(u.id) as usage_count
                    FROM asset_cache a
                    LEFT JOIN usage_stats u ON a.asset_id = u.asset_id
                    GROUP BY a.asset_id
                    ORDER BY usage_count DESC
                    LIMIT ?
                """
                params = (limit,)
            
            results = []
            for row in conn.execute(query, params):
                asset = self._row_to_asset(row)
                usage_count = row['usage_count']
                results.append((asset, usage_count))
            
            return results
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of entries removed
        """
        with self._get_connection() as conn:
            # Remove expired queries
            cursor = conn.execute("""
                DELETE FROM query_cache 
                WHERE expires_at < datetime('now')
            """)
            deleted_queries = cursor.rowcount
            
            # Remove old unused assets (not accessed in 30 days)
            cursor = conn.execute("""
                DELETE FROM asset_cache
                WHERE last_accessed < datetime('now', '-30 days')
                AND asset_id NOT IN (
                    SELECT canonical_asset_id FROM canonical_entities
                )
            """)
            deleted_assets = cursor.rowcount
            
            return deleted_queries + deleted_assets
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_connection() as conn:
            stats = {}
            
            # Query cache stats
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    COUNT(CASE WHEN expires_at > datetime('now') THEN 1 END) as active_queries,
                    SUM(hit_count) as total_hits
                FROM query_cache
            """).fetchone()
            
            stats['query_cache'] = {
                'total': row['total_queries'],
                'active': row['active_queries'],
                'hits': row['total_hits'] or 0
            }
            
            # Asset cache stats
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_assets,
                    COUNT(DISTINCT source) as sources,
                    SUM(access_count) as total_accesses
                FROM asset_cache
            """).fetchone()
            
            stats['asset_cache'] = {
                'total': row['total_assets'],
                'sources': row['sources'],
                'accesses': row['total_accesses'] or 0
            }
            
            # Entity stats
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_entities,
                    COUNT(DISTINCT entity_type) as entity_types
                FROM canonical_entities
            """).fetchone()
            
            stats['entities'] = {
                'total': row['total_entities'],
                'types': row['entity_types']
            }
            
            # Performance metrics
            stats['performance'] = self.metrics.copy()
            if self.metrics['hits'] + self.metrics['misses'] > 0:
                stats['performance']['hit_rate'] = (
                    self.metrics['hits'] / 
                    (self.metrics['hits'] + self.metrics['misses'])
                )
            else:
                stats['performance']['hit_rate'] = 0.0
            
            return stats
    
    def _hash_query(self, query: str, adapter: str) -> str:
        """Create hash for query + adapter combination."""
        combined = f"{adapter}:{query}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
    
    def _hash_file(self, file_path: str) -> Optional[str]:
        """Calculate SHA-256 hash of a file."""
        if not file_path or not os.path.exists(file_path):
            return None
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _serialize_asset(self, asset: Asset) -> Dict[str, Any]:
        """Serialize Asset object to dictionary."""
        return {
            'id': asset.id,
            'url': asset.url,
            'type': asset.type,
            'source': asset.source,
            'licence': asset.licence,
            'attribution': asset.attribution,
            'dimensions': list(asset.dimensions),
            'duration': asset.duration,
            'relevance_score': asset.relevance_score,
            'quality_score': asset.quality_score,
            'diversity_score': asset.diversity_score,
            'local_path': asset.local_path,
            'metadata': asset.metadata
        }
    
    def _deserialize_asset(self, data: Dict[str, Any]) -> Asset:
        """Deserialize dictionary to Asset object."""
        return Asset(
            id=data['id'],
            url=data['url'],
            type=data['type'],
            source=data['source'],
            licence=data['licence'],
            attribution=data.get('attribution'),
            dimensions=tuple(data['dimensions']),
            duration=data.get('duration'),
            relevance_score=data.get('relevance_score', 0.0),
            quality_score=data.get('quality_score', 0.0),
            diversity_score=data.get('diversity_score', 0.0),
            local_path=data.get('local_path'),
            metadata=data.get('metadata', {})
        )
    
    def _row_to_asset(self, row: sqlite3.Row) -> Asset:
        """Convert database row to Asset object."""
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return Asset(
            id=row['asset_id'],
            url=row['url'],
            type=metadata.get('type', 'image'),
            source=row['source'],
            licence=row['license'] or 'unknown',
            attribution=metadata.get('attribution'),
            dimensions=(row['width'] or 0, row['height'] or 0),
            duration=row['duration'],
            local_path=row['local_path'],
            metadata=metadata
        )


# Global cache instance
_cache = None


def get_cache(cache_dir: Optional[str] = None) -> AdvancedCache:
    """Get or create global cache instance."""
    global _cache
    if _cache is None:
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), 'cache', 'visual_director')
        _cache = AdvancedCache(cache_dir)
    return _cache
