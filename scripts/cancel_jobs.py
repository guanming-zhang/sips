import subprocess

# #cancelling multipe jobs
# job_id_end = 57329799
# job_id_start = 57329814
# for job_id in range(job_id_start,job_id_end):
#     subprocess.run(["scancel",str(job_id)])

#cancelling 1 job
job_id = 63612261
subprocess.run(["scancel",str(job_id)])

