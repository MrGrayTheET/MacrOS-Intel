"""
Unified Async File Manager for High-Performance Data Operations

This module provides a centralized, asynchronous file management system that:
- Handles both HDF5 and CSV file operations concurrently
- Manages write queues to prevent file locks and conflicts
- Provides batch operations for improved performance
- Offers progress tracking and error handling
- Supports metadata management alongside data files
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
import pandas as pd
# import aiofiles  # Not needed for HDF5 operations
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time
import h5py


@dataclass
class FileOperation:
    """Represents a file operation to be executed"""
    operation_type: str  # 'hdf_write', 'csv_write', 'hdf_read', 'csv_read'
    file_path: str
    key: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None
    priority: int = 0  # Higher numbers = higher priority
    timestamp: datetime = field(default_factory=datetime.now)
    operation_id: str = field(default_factory=lambda: f"op_{int(time.time() * 1000000)}")


@dataclass
class ProgressTracker:
    """Tracks progress of file operations"""
    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def progress_percentage(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return (self.completed_operations / self.total_operations) * 100
    
    @property
    def elapsed_time(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def operations_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.completed_operations / self.elapsed_time


class AsyncFileManager:
    """
    Unified async file manager for high-performance data operations.
    
    Features:
    - Concurrent file operations with semaphore control
    - Write queue management to prevent conflicts
    - Batch operations for improved efficiency
    - Progress tracking and logging
    - Error handling and retry logic
    """
    
    def __init__(self, 
                 max_concurrent_operations: int = 5,
                 max_workers: int = 4,
                 enable_logging: bool = True):
        
        self.max_concurrent_operations = max_concurrent_operations
        self.max_workers = max_workers
        self.enable_logging = enable_logging
        
        # Async semaphore for controlling concurrent operations
        self.semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Operation queue and tracking
        self.operation_queue: asyncio.Queue = asyncio.Queue()
        self.active_operations: Dict[str, FileOperation] = {}
        self.completed_operations: Dict[str, FileOperation] = {}
        self.failed_operations: Dict[str, Tuple[FileOperation, Exception]] = {}
        
        # Progress tracking
        self.progress_tracker = ProgressTracker()
        
        # File locks to prevent conflicts
        self.file_locks: Dict[str, asyncio.Lock] = {}
        
        # Setup logging
        if enable_logging:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = None
    
    def _log(self, level: str, message: str):
        """Internal logging method"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
    
    async def _get_file_lock(self, file_path: str) -> asyncio.Lock:
        """Get or create a lock for the given file path"""
        if file_path not in self.file_locks:
            self.file_locks[file_path] = asyncio.Lock()
        return self.file_locks[file_path]
    
    def _hdf_write_sync(self, file_path: str, key: str, data: pd.DataFrame, 
                       metadata: Optional[Dict] = None) -> None:
        """Synchronous HDF5 write operation"""
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write data - use fixed format to support MultiIndex columns
            data.to_hdf(file_path, key=key, mode='a', format='fixed')
            
            # Write metadata if provided
            if metadata:
                metadata_key = f"{key}_metadata"
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_hdf(file_path, key=metadata_key, mode='a', format='table')
                
        except Exception as e:
            raise Exception(f"HDF5 write failed for {file_path}:{key} - {str(e)}")
    
    def _hdf_read_sync(self, file_path: str, key: str) -> pd.DataFrame:
        """Synchronous HDF5 read operation"""
        try:
            return pd.read_hdf(file_path, key=key)
        except Exception as e:
            raise Exception(f"HDF5 read failed for {file_path}:{key} - {str(e)}")
    
    def _csv_write_sync(self, file_path: str, data: pd.DataFrame) -> None:
        """Synchronous CSV write operation"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(file_path, index=False)
        except Exception as e:
            raise Exception(f"CSV write failed for {file_path} - {str(e)}")
    
    def _csv_read_sync(self, file_path: str) -> pd.DataFrame:
        """Synchronous CSV read operation"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Exception(f"CSV read failed for {file_path} - {str(e)}")
    
    async def _execute_operation(self, operation: FileOperation) -> Any:
        """Execute a single file operation"""
        async with self.semaphore:
            file_lock = await self._get_file_lock(operation.file_path)
            
            async with file_lock:
                self._log("info", f"Executing {operation.operation_type} for {operation.operation_id}")
                
                try:
                    loop = asyncio.get_event_loop()
                    
                    if operation.operation_type == 'hdf_write':
                        result = await loop.run_in_executor(
                            self.executor,
                            self._hdf_write_sync,
                            operation.file_path,
                            operation.key,
                            operation.data,
                            operation.metadata
                        )
                        
                    elif operation.operation_type == 'hdf_read':
                        result = await loop.run_in_executor(
                            self.executor,
                            self._hdf_read_sync,
                            operation.file_path,
                            operation.key
                        )
                        
                    elif operation.operation_type == 'csv_write':
                        result = await loop.run_in_executor(
                            self.executor,
                            self._csv_write_sync,
                            operation.file_path,
                            operation.data
                        )
                        
                    elif operation.operation_type == 'csv_read':
                        result = await loop.run_in_executor(
                            self.executor,
                            self._csv_read_sync,
                            operation.file_path
                        )
                        
                    else:
                        raise ValueError(f"Unknown operation type: {operation.operation_type}")
                    
                    # Execute callback if provided
                    if operation.callback:
                        if asyncio.iscoroutinefunction(operation.callback):
                            await operation.callback(result, operation)
                        else:
                            operation.callback(result, operation)
                    
                    # Update tracking
                    self.completed_operations[operation.operation_id] = operation
                    self.progress_tracker.completed_operations += 1
                    
                    self._log("info", f"Completed {operation.operation_type} for {operation.operation_id}")
                    return result
                    
                except Exception as e:
                    self.failed_operations[operation.operation_id] = (operation, e)
                    self.progress_tracker.failed_operations += 1
                    self._log("error", f"Failed {operation.operation_type} for {operation.operation_id}: {str(e)}")
                    raise e
                finally:
                    # Remove from active operations
                    self.active_operations.pop(operation.operation_id, None)
    
    async def queue_operation(self, operation: FileOperation) -> str:
        """Queue a file operation for execution"""
        self.progress_tracker.total_operations += 1
        self.active_operations[operation.operation_id] = operation
        await self.operation_queue.put(operation)
        return operation.operation_id
    
    async def execute_queued_operations(self, max_concurrent: Optional[int] = None) -> Dict[str, Any]:
        """Execute all queued operations concurrently"""
        if max_concurrent is None:
            max_concurrent = self.max_concurrent_operations
        
        operations = []
        while not self.operation_queue.empty():
            try:
                operation = self.operation_queue.get_nowait()
                operations.append(operation)
            except asyncio.QueueEmpty:
                break
        
        if not operations:
            return {"completed": 0, "failed": 0, "results": []}
        
        # Sort by priority (higher first)
        operations.sort(key=lambda x: x.priority, reverse=True)
        
        self._log("info", f"Executing {len(operations)} queued operations")
        
        # Create semaphore for batch operations
        batch_semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(op):
            async with batch_semaphore:
                return await self._execute_operation(op)
        
        # Execute operations concurrently
        tasks = [execute_with_semaphore(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compile results
        completed_count = sum(1 for r in results if not isinstance(r, Exception))
        failed_count = len(results) - completed_count
        
        return {
            "completed": completed_count,
            "failed": failed_count,
            "results": results,
            "operations": operations
        }
    
    async def write_hdf(self, file_path: str, key: str, data: pd.DataFrame, 
                       metadata: Optional[Dict] = None, 
                       callback: Optional[Callable] = None,
                       priority: int = 0) -> str:
        """Queue HDF5 write operation"""
        operation = FileOperation(
            operation_type='hdf_write',
            file_path=file_path,
            key=key,
            data=data,
            metadata=metadata,
            callback=callback,
            priority=priority
        )
        return await self.queue_operation(operation)
    
    async def read_hdf(self, file_path: str, key: str, 
                      callback: Optional[Callable] = None,
                      priority: int = 0) -> str:
        """Queue HDF5 read operation"""
        operation = FileOperation(
            operation_type='hdf_read',
            file_path=file_path,
            key=key,
            callback=callback,
            priority=priority
        )
        return await self.queue_operation(operation)
    
    async def write_csv(self, file_path: str, data: pd.DataFrame,
                       callback: Optional[Callable] = None,
                       priority: int = 0) -> str:
        """Queue CSV write operation"""
        operation = FileOperation(
            operation_type='csv_write',
            file_path=file_path,
            data=data,
            callback=callback,
            priority=priority
        )
        return await self.queue_operation(operation)
    
    async def read_csv(self, file_path: str,
                      callback: Optional[Callable] = None,
                      priority: int = 0) -> str:
        """Queue CSV read operation"""
        operation = FileOperation(
            operation_type='csv_read',
            file_path=file_path,
            callback=callback,
            priority=priority
        )
        return await self.queue_operation(operation)
    
    async def batch_write_hdf(self, operations: List[Dict[str, Any]], 
                             execute_immediately: bool = True) -> Dict[str, Any]:
        """Queue multiple HDF5 write operations"""
        operation_ids = []
        
        for op_data in operations:
            operation = FileOperation(
                operation_type='hdf_write',
                file_path=op_data['file_path'],
                key=op_data['key'],
                data=op_data['data'],
                metadata=op_data.get('metadata'),
                callback=op_data.get('callback'),
                priority=op_data.get('priority', 0)
            )
            operation_id = await self.queue_operation(operation)
            operation_ids.append(operation_id)
        
        if execute_immediately:
            results = await self.execute_queued_operations()
            return {**results, "operation_ids": operation_ids}
        
        return {"operation_ids": operation_ids}
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            "total_operations": self.progress_tracker.total_operations,
            "completed_operations": self.progress_tracker.completed_operations,
            "failed_operations": self.progress_tracker.failed_operations,
            "progress_percentage": self.progress_tracker.progress_percentage,
            "elapsed_time": self.progress_tracker.elapsed_time,
            "operations_per_second": self.progress_tracker.operations_per_second,
            "active_operations": len(self.active_operations),
            "queued_operations": self.operation_queue.qsize()
        }
    
    def get_failed_operations(self) -> Dict[str, Tuple[FileOperation, Exception]]:
        """Get information about failed operations"""
        return self.failed_operations.copy()
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all operations to complete"""
        start_time = time.time()
        
        while (self.active_operations or not self.operation_queue.empty()):
            if timeout and (time.time() - start_time) > timeout:
                return False
            await asyncio.sleep(0.1)
        
        return True
    
    async def shutdown(self):
        """Shutdown the file manager and cleanup resources"""
        self._log("info", "Shutting down AsyncFileManager")
        
        # Wait for pending operations
        await self.wait_for_completion(timeout=30.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self._log("info", "AsyncFileManager shutdown complete")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()


# Convenience functions for common operations
async def async_hdf_batch_write(file_path: str, 
                               data_dict: Dict[str, pd.DataFrame],
                               metadata_dict: Optional[Dict[str, Dict]] = None,
                               max_concurrent: int = 5) -> Dict[str, Any]:
    """
    Write multiple DataFrames to HDF5 file concurrently
    
    Args:
        file_path: Path to HDF5 file
        data_dict: Dictionary of {key: DataFrame}
        metadata_dict: Optional dictionary of {key: metadata_dict}
        max_concurrent: Maximum concurrent operations
    
    Returns:
        Dictionary with operation results
    """
    async with AsyncFileManager(max_concurrent_operations=max_concurrent) as manager:
        operations = []
        
        for key, data in data_dict.items():
            metadata = metadata_dict.get(key) if metadata_dict else None
            operations.append({
                'file_path': file_path,
                'key': key,
                'data': data,
                'metadata': metadata
            })
        
        return await manager.batch_write_hdf(operations, execute_immediately=True)


async def async_progress_callback(result: Any, operation: FileOperation):
    """Example progress callback function"""
    print(f"Operation {operation.operation_id} completed: {operation.operation_type}")


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    async def test_async_file_manager():
        # Create test data
        test_data = {
            'test/data1': pd.DataFrame({'col1': range(100), 'col2': np.random.randn(100)}),
            'test/data2': pd.DataFrame({'col1': range(200), 'col2': np.random.randn(200)}),
            'test/data3': pd.DataFrame({'col1': range(300), 'col2': np.random.randn(300)})
        }
        
        test_file = "test_async_operations.h5"
        
        print("Testing AsyncFileManager...")
        
        # Test batch operations
        result = await async_hdf_batch_write(
            file_path=test_file,
            data_dict=test_data,
            max_concurrent=3
        )
        
        print(f"Batch write completed: {result['completed']} successful, {result['failed']} failed")
        
        # Test individual operations with manager
        async with AsyncFileManager(max_concurrent_operations=3) as manager:
            # Queue some read operations
            read_ids = []
            for key in test_data.keys():
                read_id = await manager.read_hdf(
                    file_path=test_file,
                    key=key,
                    callback=async_progress_callback
                )
                read_ids.append(read_id)
            
            # Execute all queued operations
            read_results = await manager.execute_queued_operations()
            print(f"Read operations: {read_results['completed']} completed, {read_results['failed']} failed")
            
            # Show progress
            progress = manager.get_progress()
            print(f"Progress: {progress['progress_percentage']:.1f}% complete")
            print(f"Speed: {progress['operations_per_second']:.1f} ops/sec")
    
    # Run the test
    asyncio.run(test_async_file_manager())