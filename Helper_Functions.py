
# Helper Functions

def unique_jobs_count(jobs):
    job_dict=dict()
    for x in jobs:
        job_dict[x[0]] = ''
    return len(job_dict)


def inflation(value, year):
    inflation = 1
    if year == 2011: inflation = 1.07;
    if year == 2012: inflation = 1.05;
    if year == 2013: inflation = 1.04;
    if year == 2014: inflation = 1.02;
    return inflation * value


def status_calc(status, pay):
    if status == "" or status == None:
        return ("pt" if pay < 50000 else "ft")
    return status


def total_pay_converter(pay):
    if pay < 100: return pay * 40 * 52
    elif pay < 1000: return pay * 26
    return pay


def encode_prediction(val, X1enc, X1dec, X2enc, X2dec):
    job = ''
    status = ''
    for x in range(len(X1dec)):
        if val[0].lower() == X1dec[x]:
            job = X1enc[x]
        if val[1].lower() == X2dec[x]:
            status = X2enc[x]
    if job == '': job = max(X1enc) + 1
    if status == '': status = max(X2enc) + 1
return [job, status]