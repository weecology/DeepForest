---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is. For questions and community discussion, please create a discussion (https://github.com/weecology/DeepForest/discussions). 

**To Reproduce**
Issues without reproducible code that we can run to explore the problem are much more difficult to understand and debug and so are much less likely to be addressed quickly.

```
# Load the modules
from deepforest import main
from deepforest import get_data
import os

# Use the latest release
m = main.deepforest()
m.use_release()

# Use package data for simple training example
m.config["train"]["csv_file"] = get_data("example.csv") 
m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
m.config["train"]["fast_dev_run"] = True    
m.trainer.fit(m)
```

**Environment (please complete the following information):**
 - OS: 
 - Python version and environment : 

**Screenshots and Context**
If applicable, add screenshots to help explain your problem. Please paste entire code instead of a snippet! 

**User Story**
Tell us about who you are and what you hope to achieve with DeepForest

“As a [type of user] I want [my goal] so that [my reason].”


