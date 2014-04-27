int main(void) {
    macro unless {
        case {
            match {
                $(cond)
            }
            template {
                if(! $(cond))
            }
        }
    }

    unless(true) {
        printf("In 'unless'\n");
    }

    unless(false) {
        printf("Should never run!\n");
    }
}
