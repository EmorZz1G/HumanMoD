# python main.py --doctor_num 3 --use_rag 1 --use_assi 1 --use_senior 1 --use_correct 1 --rag_top_k 1 --rag_opt_top_k 1 --version 'ours' --api_port 8001

# python main.py --doctor_num 3 --use_rag 1 --use_assi 1 --use_senior 1 --use_correct 1 --rag_top_k 1 --rag_opt_top_k 1 --version 'ours' --api_port 8000 --test_file_pth './datas/tests/US_test.jsonl'

# python main.py --doctor_num 1 --use_rag 1 --use_assi 1 --use_senior 1 --use_correct 1 --rag_top_k 1 --rag_opt_top_k 1 --version 'ours_N1_sel' --api_port 8005 --test_file_pth './datas/tests/US_test.jsonl'

# python main.py --doctor_num 3 --use_rag 1 --use_assi 1 --use_senior 1 --use_correct 1 --rag_top_k 1 --rag_opt_top_k 0 --version 'ours_N3_only1_rag_opt0' --api_port 8005 --test_file_pth './datas/tests/US_test.jsonl' --doctor_promts_pth './prompts/doctors2/'

# python main.py --doctor_num 3 --use_rag 1 --use_assi 1 --use_senior 1 --use_correct 1 --rag_top_k 1 --rag_opt_top_k 0 --version 'ours_N3_rag_opt0' --api_port 8005 --test_file_pth './datas/tests/US_test.jsonl'

python main.py --doctor_num 3 --use_rag 1 --use_assi 1 --use_senior 1 --use_correct 1 --rag_top_k 1 --rag_opt_top_k 0 --version 'ours_N3_only1_rag_opt0_true' --api_port 8005 --test_file_pth './datas/tests/US_test.jsonl' --doctor_promts_pth './prompts/doctors3/'

python main.py --doctor_num 3 --use_rag 1 --use_assi 1 --use_senior 1 --use_correct 1 --rag_top_k 1 --rag_opt_top_k 1 --version 'ours_N3_only1_w_opt_rag' --api_port 8005 --test_file_pth './datas/tests/US_test.jsonl' --doctor_promts_pth './prompts/doctors3/'