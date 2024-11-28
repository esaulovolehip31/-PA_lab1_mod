import os
import heapq
import numpy as np
import time

CHUNK_SIZE_MB = 100  # Розмір однієї серії в мегабайтах


def generate_large_file(filename, size_in_mb):
    """
    Генерує великий файл із випадковими числами.
    """
    num_elements = (size_in_mb * 1024 * 1024) // 4  # 1 число ≈ 4 байти в бінарному вигляді
    with open(filename, "wb") as f:
        data = np.random.randint(0, 10 ** 6, size=num_elements, dtype=np.int32)
        data.tofile(f)


def split_into_sorted_chunks(input_file, chunk_size_mb):
    """
    Розбиває великий файл на відсортовані серії.
    """
    chunk_files = []
    chunk_size = (chunk_size_mb * 1024 * 1024) // 4  # Кількість чисел у серії
    with open(input_file, "rb") as f:
        while True:
            data = np.fromfile(f, dtype=np.int32, count=chunk_size)
            if data.size == 0:
                break
            data.sort()
            chunk_index = len(chunk_files) + 1
            chunk_file = f"chunk_{chunk_index}.bin"
            with open(chunk_file, "wb") as chunk_f:
                data.tofile(chunk_f)
            chunk_files.append(chunk_file)
    return chunk_files


def merge_sorted_chunks(chunk_files, output_file):
    """
    Зливає відсортовані серії у один вихідний файл.
    """
    file_pointers = [open(chunk, "rb") for chunk in chunk_files]
    heap = []
    buffers = [np.empty(1024, dtype=np.int32) for _ in file_pointers]

    for i, fp in enumerate(file_pointers):
        data = np.fromfile(fp, dtype=np.int32, count=len(buffers[i]))
        if data.size > 0:
            buffers[i] = data
            heapq.heappush(heap, (buffers[i][0], i, 0))

    with open(output_file, "wb") as out_f:
        while heap:
            min_value, file_index, buffer_index = heapq.heappop(heap)
            out_f.write(np.array([min_value], dtype=np.int32).tobytes())
            buffer_index += 1
            if buffer_index < len(buffers[file_index]):
                heapq.heappush(heap, (buffers[file_index][buffer_index], file_index, buffer_index))
            else:
                data = np.fromfile(file_pointers[file_index], dtype=np.int32, count=len(buffers[file_index]))
                if data.size > 0:
                    buffers[file_index] = data
                    heapq.heappush(heap, (buffers[file_index][0], file_index, 0))

    for fp in file_pointers:
        fp.close()
    for chunk in chunk_files:
        os.remove(chunk)


if __name__ == "__main__":
    INPUT_FILE = "large_input.bin"
    OUTPUT_FILE = "sorted_output.bin"
    FILE_SIZE_MB = 1024  # Розмір файлу в мб

    # Генерація великого файлу
    print("Генерація великого файлу...")
    start_time = time.time()
    generate_large_file(INPUT_FILE, FILE_SIZE_MB)
    generate_time = time.time() - start_time
    print(f"Файл '{INPUT_FILE}' створено. Час генерації: {generate_time:.2f} секунд.\n")

    # Розбиття на серії
    print("Розбиття на серії...")
    start_time = time.time()
    chunk_files = split_into_sorted_chunks(INPUT_FILE, CHUNK_SIZE_MB)
    split_time = time.time() - start_time
    print(f"Розбито на {len(chunk_files)} серій. Час розбиття: {split_time:.2f} секунд.\n")

    # Злиття серій
    print("Злиття серій...")
    start_time = time.time()
    merge_sorted_chunks(chunk_files, OUTPUT_FILE)
    merge_time = time.time() - start_time
    print(f"Сортування завершено. Вихідний файл: '{OUTPUT_FILE}'. Час злиття: {merge_time:.2f} секунд.\n")

    # Загальний час
    total_time = generate_time + split_time + merge_time
    print(f"Загальний час виконання: {total_time:.2f} секунд.")

