import multiprocessing
import time

def sleepy_man(sec, return_dic):
    return_dic[sec] = f"lol {str(sec)}"


if __name__ == "__main__":
  tic = time.time()
  manager = multiprocessing.Manager()
  my_dict = manager.dict()
  pool = multiprocessing.Pool(5)
  my_args = []
  for x in range(1,11):
    my_args.append((x, my_dict))
  pool.starmap(sleepy_man, my_args)
  pool.close()

  print(my_dict)

  toc = time.time()