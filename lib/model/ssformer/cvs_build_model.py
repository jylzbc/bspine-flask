def build_cvs(model_name, class_num=5):
    if model_name == "mit_PLD_b2_cvs":
        from .mit.mit_PLD_b2_cvs import mit_PLD_b2
        model = mit_PLD_b2(class_num=class_num)
        return model    
