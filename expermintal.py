import time
import sys
import psutil
import os
import gc

def get_memory_usage():
    """获取当前进程的内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_large_list(size):
    """创建一个大列表"""
    return [str(i) * 100 for i in range(size)]

def test_assignment_clear(size, iterations):
    """测试使用 list = [] 清空列表的性能"""
    print(f"\n=== 测试 list = [] (列表大小: {size}, 迭代: {iterations}) ===")
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # 创建初始列表
    my_list = create_large_list(size)
    
    for i in range(iterations):
        # 使用赋值方式清空列表
        my_list = []
        
        # 重新填充列表
        my_list.extend(create_large_list(size))
        
        # 每完成一定比例的迭代后显示进度
        if (i + 1) % max(1, iterations // 10) == 0:
            current_memory = get_memory_usage()
            print(f"进度: {i + 1}/{iterations}, 当前内存: {current_memory:.2f} MB")
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    time_used = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"时间: {time_used:.4f} 秒")
    print(f"内存使用变化: {memory_used:.2f} MB")
    
    return time_used, memory_used

def test_clear_method(size, iterations):
    """测试使用 list.clear() 清空列表的性能"""
    print(f"\n=== 测试 list.clear() (列表大小: {size}, 迭代: {iterations}) ===")
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # 创建初始列表
    my_list = create_large_list(size)
    
    for i in range(iterations):
        # 使用 clear 方法清空列表
        my_list.clear()
        
        # 重新填充列表
        my_list.extend(create_large_list(size))
        
        # 每完成一定比例的迭代后显示进度
        if (i + 1) % max(1, iterations // 10) == 0:
            current_memory = get_memory_usage()
            print(f"进度: {i + 1}/{iterations}, 当前内存: {current_memory:.2f} MB")
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    time_used = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"时间: {time_used:.4f} 秒")
    print(f"内存使用变化: {memory_used:.2f} MB")
    
    return time_used, memory_used

def test_memory_release():
    """测试内存释放情况"""
    print("\n=== 测试内存释放情况 ===")
    
    # 测试 list = []
    gc.collect()  # 强制垃圾回收
    initial_memory = get_memory_usage()
    
    large_list = create_large_list(100000)
    after_creation = get_memory_usage()
    
    large_list = []  # 使用赋值清空
    gc.collect()  # 强制垃圾回收
    after_assignment = get_memory_usage()
    
    print(f"list = [] 内存变化:")
    print(f"  创建后: {after_creation - initial_memory:.2f} MB")
    print(f"  清空后: {after_assignment - initial_memory:.2f} MB")
    print(f"  释放了: {(after_creation - after_assignment):.2f} MB")
    
    # 测试 list.clear()
    gc.collect()  # 强制垃圾回收
    initial_memory = get_memory_usage()
    
    large_list = create_large_list(100000)
    after_creation = get_memory_usage()
    
    large_list.clear()  # 使用 clear 方法清空
    gc.collect()  # 强制垃圾回收
    after_clear = get_memory_usage()
    
    print(f"list.clear() 内存变化:")
    print(f"  创建后: {after_creation - initial_memory:.2f} MB")
    print(f"  清空后: {after_clear - initial_memory:.2f} MB")
    print(f"  释放了: {(after_creation - after_clear):.2f} MB")

def test_reference_behavior():
    """测试引用行为差异"""
    print("\n=== 测试引用行为差异 ===")
    
    # 测试 list = [] 的引用行为
    print("list = [] 的引用行为:")
    original_list = [1, 2, 3, 4, 5]
    reference = original_list  # 创建引用
    print(f"原始列表: {original_list}, 引用: {reference}")
    
    original_list = []  # 重新赋值
    print(f"重新赋值后 - 原始列表: {original_list}, 引用: {reference}")
    print("引用仍然指向原来的列表对象")
    
    # 测试 list.clear() 的引用行为
    print("\nlist.clear() 的引用行为:")
    original_list = [1, 2, 3, 4, 5]
    reference = original_list  # 创建引用
    print(f"原始列表: {original_list}, 引用: {reference}")
    
    original_list.clear()  # 清空列表
    print(f"清空后 - 原始列表: {original_list}, 引用: {reference}")
    print("引用和原始列表都指向同一个已清空的对象")

def test_different_sizes():
    """测试不同列表大小下的性能"""
    print("\n" + "="*60)
    print("测试不同列表大小下的性能")
    print("="*60)
    
    sizes = [1000, 10000, 100000]
    iterations = 100
    
    results = {}
    
    for size in sizes:
        print(f"\n>>> 测试列表大小: {size}")
        assignment_time, assignment_memory = test_assignment_clear(size, iterations)
        clear_time, clear_memory = test_clear_method(size, iterations)
        
        results[size] = {
            'assignment': (assignment_time, assignment_memory),
            'clear': (clear_time, clear_memory)
        }
    
    # 显示比较结果
    print("\n" + "="*60)
    print("不同大小列表的性能比较:")
    print("="*60)
    
    print(f"{'列表大小':<12} {'方法':<15} {'时间(秒)':<12} {'内存变化(MB)':<15}")
    print("-" * 60)
    
    for size in sizes:
        assignment_time, assignment_memory = results[size]['assignment']
        clear_time, clear_memory = results[size]['clear']
        
        print(f"{size:<12} {'list = []':<15} {assignment_time:<12.4f} {assignment_memory:<15.2f}")
        print(f"{size:<12} {'list.clear()':<15} {clear_time:<12.4f} {clear_memory:<15.2f}")

def main():
    """主测试函数"""
    print("Python列表清空方法性能比较: list = [] vs list.clear()")
    print("注意: 需要安装 psutil: pip install psutil")
    
    # 测试基本性能
    size = 50000
    iterations = 50
    
    assignment_time, assignment_memory = test_assignment_clear(size, iterations)
    clear_time, clear_memory = test_clear_method(size, iterations)
    
    # 显示比较结果
    print("\n" + "="*60)
    print("性能比较总结:")
    print("="*60)
    
    print(f"{'方法':<15} {'时间(秒)':<12} {'内存变化(MB)':<15}")
    print("-" * 45)
    print(f"{'list = []':<15} {assignment_time:<12.4f} {assignment_memory:<15.2f}")
    print(f"{'list.clear()':<15} {clear_time:<12.4f} {clear_memory:<15.2f}")
    
    # 计算性能差异
    time_diff = assignment_time - clear_time
    time_percent = (time_diff / clear_time) * 100 if clear_time > 0 else 0
    
    print(f"\n时间差异: {time_diff:.4f} 秒 ({time_percent:+.2f}%)")
    
    if time_diff > 0:
        print("list.clear() 更快")
    else:
        print("list = [] 更快")
    
    # 测试其他方面
    test_memory_release()
    test_reference_behavior()
    test_different_sizes()

if __name__ == "__main__":
    # 确保垃圾回收在测试前是开启的
    gc.enable()
    main()