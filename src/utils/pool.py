import multiprocessing
CPUs=1 # XXX set here default number of CPUs for multiprocessing

def apply_on_list(lst, fnc):
    ret = []
    for x in lst:
        try:
            ret.append(fnc(x))
        except:
            print(x,'in',lst)
            ret.append(x) # BUG error occurs when F[H-]F is converted into fragSMILES: full SMILES representation is retained
    return ret

def applyFncPool(column, fnc, cpus=CPUs):
    with multiprocessing.Pool(processes=cpus) as pool:
        column = pool.map(fnc, column)

    return column