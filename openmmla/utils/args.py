import distutils.util


def print_arguments(args):
    print("-----------  Input Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))


def add_arguments(argname, type, default, help, argparser, shortname=None, **kwargs):
    # 处理 bool 类型（转成 distutils.util.strtobool）
    internal_type = distutils.util.strtobool if type == bool else type
    type_str = "bool" if type == bool else type.__name__

    # 添加短名和长名参数
    args = ["--" + argname]
    if shortname:
        args.insert(0, shortname)

    # 加上类型说明 & 默认值
    help += f" (type: {type_str}, default: {default})"

    argparser.add_argument(
        *args,
        default=default,
        type=internal_type,
        help=help,
        metavar=argname,  # 控制 help 输出小写，不是全大写
        **kwargs
    )
