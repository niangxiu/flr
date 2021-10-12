run :
	@rm -f *p *png log
	@rm -rf __pycache__
	python3 -u plot.py | tee log

clean :
	@rm -f *p *png log
	@rm -rf __pycache__

