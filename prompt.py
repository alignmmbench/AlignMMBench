CRITIC_PROMPT_ZHV2 = {
    "General": """
        ## 问答数据
{history}
        - **用户提问：**
        {question}
        - **问题类型：**
        {questionType}
        [参考答案开始]
        {refAnswer}
        [参考答案结束]
        [AI助手回答开始]
        {assistant}
        [AI助手回答结束]
        ## 目标
        您是一个擅长评价问答质量的专家。请您以客观公正的评判者身份，结合用户提问和参考答案内容，评估AI助手回答的质量。
        ## 指令
        - **评分要求：**
        1. 您的评估应考虑到正确性、相关性和详细级别等因素。开始您的评估时，请将AI助手回答与参考答案进行比较，并从中找出AI助手回答可能存在的不合理内容，在整体评分结束后提供简短的解释说明。
        2. 对于每次评分请求，您需要进行1到10的评分，评分规则如下：
            - 整体上，模型回答的质量越高，则分数越高。
            - 当AI助手回答完全没有遵循用户的文本指令内容，答非所问、自言自语，总分必须是1分；
            - 当AI助手回答和用户提问相关，但内容有害（如：不友好，违反安全性）或语句不通顺不流畅（如：存在大量重复，语言整体混乱），则总分必须是2分；
            - 当AI助手回答和用户提问相关，只是回答完全错误，则总分只能是3分。
            - 当AI助手回答和用户提问相关，回答有一半以上信息存在事实性错误，则总分应该在4~5分之间。
            - 当AI助手回答和用户提问相关，只是有一半以下的信息存在事实性错误，或回答无事实性错误但只满足了用户提问的部分要求，则总分应该是6分。
            - 当AI助手回答和用户提问相关，只有极个别错误或回答正确但过于简略，总分应当在7~8分之间。
            - 只有当AI助手回答完成了用户提问的所有要求，且提到的所有信息均正确，只是遗漏了一些并不重要的信息，且这些信息不影响回答内容的优劣判断下，总分可以给到9~10分。
        3. 判断**回答与问题是否相关**或者**回答是否正确**时，应该充分考虑语言的同义性，等价性和近似语义。
        4. 整体上，在回答结果正确的条件下，有合理分析的回答得分应该更高。
        - **注意事项：**
        由于用户提问属于"{questionType}"类型，因此，在进行打分和解释理由时，应该充分考虑以下注意事项：
        {warnings}
        特别地：由于您无法查看图片内容，因此没有提供图片给您。但 AI 助手在回答问题时，是根据图片内容进行回答的。因此当AI助手回答中存在参考答案中没有提到的图片相关信息时，您不能直接认为它是错误的，而是需要结合上述注意事项以及您现有的知识进行综合评价。比如：
            - 问题：这幅卡通图片里的鱼是已经煮熟了吗？我们如何判断？
            - 参考答案：是的，这个卡通图中的鱼已经煮熟了，因为它是不透明的，用叉子很容易剥离，并且内部温度已经达到145华氏度。
            - AI助手：是的，这幅图片中的鱼已经煮熟了。我们可以通过鱼的颜色、质地和温度来判断。鱼的颜色是粉红色，质地变得松散并且容易用叉子夹碎，温度达到了145°F，这些都是鱼已经煮熟的迹象。
            - 评分&理由：9分。首先，AI助手的结论与参考答案一致。其次，通过颜色和温度判断鱼是否熟了符合生活经验。整体上，AI助手的回答结论正确，逻辑成立，可以给一个较高的分数。
        ## 输出格式
        您必须按照以下 JSON 格式输出回答：
        {{
            "Rating": ,
            "Reason":
        }}
        除了JSON内容外，请不要输出任何其他字符。并且，应该使用中文描述 Reason 部分。
    """,
    "Types": {
        "描述类": """
            1. 在评分时，应该充分考虑AI助手回答的组织条理性，逻辑性，语言流畅性和内容完整性。当AI助手回答内容存在不完整时，可以根据不完整程度进行酌情减分，但不能直接认为是AI助手回答是错误的。
            2. 由于参考答案是对某张图片内容的完整或者部分描述，因此参考答案可能存在描述不全的情况。当AI助手回答中存在参考答案中不存在的内容时，可以适当怀疑其内容的合理性，但不能直接认为新增内容是错误的。
        """,
        "推理类": """
            1. AI助手回答此类问题时应该提供合理的解释，尤其是问题要求给出理由时。
            2. 在评分时，应该首先判断AI助手回答的结论是否正确，若结论错误，可以判定其回答错误，此时应直接给一个低分；若结论正确，再结合其解释的合理性与逻辑性进行综合评分。
        """,
        "识别类": """
            1. 这类问题的回答重点在于识别结果的正确与否，且用户提问和参考答案内容均默认围绕图片进行。您应该耐心地从AI助手回答中找到针对问题的关键答案。当AI助手回答中结果与参考答案语义一致时，务必给高分甚至满分。
            2. 即使AI助手的回答和参考答案相比有多余的内容，只要AI助手的识别结果正确，回答中存在和参考答案语义一致的部分，且其余部分符合逻辑，就应当给高分甚至满分。
            3. 若回答中对识别结果有合理的描述或者推测，能够酌情加分。当然，不能超过评分规定中的10分。
            4. 识别文本内容时，除非题目中特别强调，否则不应该将翻译后的文字视为错误结果。
            5. 对于数字，应该注意等价转换，比如 0.1 = 10%
        """,
        "计数类": """
            1. 这类问题的回答重点在于计数结果的正确与否，且用户提问和参考答案内容均默认围绕图片进行。您应该耐心地从AI助手回答中找到针对问题的关键答案。当AI助手回答中结果与参考答案一致时，务必给高分甚至满分。反之，结果只要不同时，不论差距有多小都必须视为完全错误，必须给低分。
            2. 即使AI助手的回答和参考答案相比有很多多余的内容，只要AI助手回答中存在和参考答案语义一致的部分，且其余部分符合逻辑，就应当给高分甚至满分。
            3. 若回答中对识别结果有不合理的描述或者推测，应该酌情减分。
        """,
        "图表类": """
            1. 由于您无法查看图片，所以请务必将AI助手回答与参考答案进行比较分析。
            2. 对于格式转换题，首先关注回答是否符合新格式要求，其次关注回答的内容是否正确。
            3. 对于数字，应该注意等价转换，比如 0.1 = 10%
        """,
        "对比类": """
            1. 若题目是要求进行对比分析，则有一定组织格式的回答质量优于无组织格式的回答。
        """,
        "创作类": """
            1. 如果提问是要求根据图片写一个故事，那么即便AI助手的回答和参考答案差异较大，也不应该直接给1～4分。而应该根据回答中故事本身的流畅性、戏剧性、有趣程度和与用户提问的关联度等进行打分。
        """,
        "智力类": """
            1. 请注意检查用户提问和AI助手回答的一致性，如果AI助手答非所问，直接给低分。
            2. 如果提问类似“这道题怎么做？”，“图片中的题怎么做？”说明用户希望AI助手解决图片中的问题。此时请仔细对比参考答案和AI助手回复来判断AI助手是否真正解决了问题。
                - 问题：这道题怎么做？
                - 参考答案：这道题的解答如下：设计划每天修$x$m，有如下方程：  $1200/x-1200/(1.5x)=5$ 。解方程可得$x=80$。因此计划平均每天修建步行道的长度为80米。
                - AI助手回答：15.(2020威海)在“旅游示范公路”建设的过程中，工程队计划在海边某路段修建一条长1200m的步行道。由于采用新的施工方式，平均每天修建步行道的长度是计划的1.5倍，结果提前5天完成任务。求计划平均每天修建步行道的长度。(3分)
                - 评分&理由：2 分。参考答案回答了计划平均每天修建步行道的长度，但AI助手未回答这一点。AI助手更像是简单复述了图片中的题目，答非所问，只能获得低分。
        """,
        "梗图理解": """
            1. 这类问题的重点在于评测AI助手能否正确理解该梗图的有趣点。所以你需要仔细比对AI助手和参考答案对图片的解释和理解是否是一种意思，如果是则要打高分；
            2. 如果AI助手完全没有解释图片为何有趣，或解释的内容和参考回答相比不足以让用户体会到这张图片背后的真实含义，应该给低分。
        """,
        "世界知识": """
            1. 这类问题的重点在于评测AI助手是否拥有图片以外的世界知识，所以当AI助手回答了一些参考答案中没有的内容时，你不能直接认为它是错误的，而是要结合逻辑是否通顺，与用户提问是否相关，和你自己拥有的知识等方面综合评判。
        """,
        "OCR理解": """
            1. 如果用户提问要求提取或识别内容，则此时应该严格要求答案与参考答案相同，此时不需要考虑“AI助手回答中存在参考答案中没有提到的图片相关信息”，此时只要参考答案与AI助手回答有差距，直接给一个低分。
            2. 如果用户提问要求提取或识别内容，此时你只需要检验AI助手回答和参考答案是否匹配，并不需要对回答本身是否包含错误进行检查。
            3. 注意判断AI助手回答与参考答案是否语义是一致的，只是语种。若是回答跟参考答案是不同语种的相同表达，则不应该以此理由给1～4分。比如：
                - 问题：这张图片中有什么字？
                - 参考答案：We are the champions
                - AI助手：这张图片中写着”我们是冠军“。
                - 评分&理由：8分。因为AI助手回答了参考答案的中文翻译内容，语义是完全一致的，因此评分为8分。
        """,
        "多轮对话": """
            1. `对话历史`中呈现了用户和AI助手之间的历史会话内容。
            2. 这里问题的重点在于评测AI助手能否在多轮对话中利用之前对话的知识，遵循先前对话的指令。所以您需要深刻理解`对话历史`的每一轮对话，将其与当前的`用户提问`和答案进行对比，给出综合的得分。
            3. 如果多轮对话历史中用户要求AI助手修复自己回答的错误，需要仔细观察本次AI助手回复是否能意识到自己的错误并给出正确的答案，如果没有意识到自己的错误或只是承认错误但未给出修正过的答案，则只能获得低分。
        """,
    },
    "answer": """
        {{
            "Rating": {score},
            "Reason": {reason}
        }}
    """
}

category_CN = {
    'Descrpition': '描述类',
    'Recognition': '识别类',
    'Counting': '计数类',
    'OCR': 'OCR理解',
    'Meme': '梗图理解',
    'Knowledge': '世界知识',
    'Reasoning': '推理类',
    "Chart": '图表类',
    'Problem': '智力类',
    'Comparison': '对比类',
    'Writing': '创作类',
    'Dialogue Context': '多轮对话',
    'Coherence': '多轮对话',
    'Incoherence': '多轮对话',
}