STUDENT_ID := NE6091051

init:
	curl https://pyenv.run | bash
	pipenv --python 3.8

run:
	pipenv run python main.py

ziphw:
	$(eval VER := $(shell cat version.txt))
	$(eval FILE_PREFIX := $(STUDENT_ID)-$(VER))
	@echo $(FILE_PREFIX)
	@mkdir $(FILE_PREFIX)
	@cp Pipfile $(FILE_PREFIX)
	@cp Pipfile.lock $(FILE_PREFIX)
	@cp main.py $(FILE_PREFIX)
	@zip $(FILE_PREFIX) $(FILE_PREFIX)/*
	@ls $(FILE_PREFIX).zip
	@rm -r $(FILE_PREFIX)

upload:
	$(eval VER := $(shell cat version.txt))
	$(eval FILE_PREFIX := $(STUDENT_ID)-$(VER))
	@echo -ne '\007'
	sftp -P 22 $(STUDENT_ID)@140.116.247.123:upload << "put $(FILE_PREFIX).zip"

submit: ziphw upload
