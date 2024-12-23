def aa():
    import subprocess
    cmd = 'arp -a | grep "d0:ef:76:ef:67:4"'
    returned_output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    parse=str(returned_output).split(' ',1)
    ip=parse[1].split(' ')
    print(ip[0][1:-1])

aa()