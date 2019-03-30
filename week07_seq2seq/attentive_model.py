from basic_model_torch import*
# inspired by https://github.com/yandexdataschool/nlp_course/tree/master/week04_seq2seq

class AttentionLayer(nn.Module):
    def __init__(self,  enc_size, dec_size, hid_size, nonlinearity=torch.nn.Tanh):
        super().__init__()
        """ A layer that computes additive attention response and weights """
        self.enc_size = enc_size # num units in encoder state
        self.dec_size = dec_size # num units in decoder state
        self.hid_size = hid_size # attention layer hidden units
        self.nonlinearity = nonlinearity()  # attention layer hidden nonlinearity

        self.lin_enc = nn.Linear(enc_size,hid_size)
        self.lin_dec = nn.Linear(dec_size,hid_size)
        self.lin_out = nn.Linear(hid_size,1)
        self.drop = nn.Dropout(p=0.6)

    def forward(self, enc, dec, mask):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, length, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param mask: mask on enc activatons (0 after first eos), float32 [batch_size, length]
        :returns: attn[batch_size, enc_size], probs[batch_size, length]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """
        dec_vec = self.drop(self.lin_dec(dec.unsqueeze(1)))
        enc_vec = self.drop(self.lin_enc(enc))

        logits = self.lin_out(self.nonlinearity(dec_vec + enc_vec)).squeeze() #[batch_size, length]
                                         #[batch_size, length, hid_size]
        # apply mask
        logits = logits*mask + -1e9*(1-mask)
        probs = torch.softmax(logits, dim=1) #[batch_size, length]
        attn = torch.sum(enc*probs.unsqueeze(-1), dim=1) #[batch_size, enc_size]
        return attn, probs



class AttentiveModel(BasicTranslationModel):
    def __init__(self, inp_voc, out_voc,
                 emb_size=64, hid_size=128, attn_size=128):
        """ Translation model that uses attention. See instructions above. """
        super().__init__(inp_voc, out_voc, emb_size, hid_size)
        del self.enc0, self.dec0
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.encoder = nn.LSTMCell(emb_size, hid_size)
        self.dec_start = nn.Linear(2*hid_size,2*hid_size)
        self.decoder = nn.LSTMCell(emb_size+hid_size, hid_size)
        self.attn = AttentionLayer(hid_size, hid_size, attn_size, torch.nn.Tanh)
        self.drop = nn.Dropout(p=0.5)


    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """

        inp_emb = self.emb_inp(inp)     # [batch_size, length, emb_size]
        state = self.encoder(inp_emb[:,0,:])  # 2x[batch_size, emb_size]
        hs = [state[0]]
        cs = [state[1]]
        for i in range(inp_emb.shape[1]-1):
            state = self.encoder(inp_emb[:,i,:], state)
            hs.append(state[0])
            cs.append(state[1])

        hs = torch.stack(hs, dim=1)
        cs = torch.stack(cs,dim=1)
        # encode, create initial decoder states
        end_index = infer_length(inp, self.inp_voc.eos_ix)
        end_index[end_index >= inp.shape[1]] = inp.shape[1] - 1
        h = hs[range(0, hs.shape[0]), end_index.detach(), :]
        c = cs[range(0, cs.shape[0]), end_index.detach(), :]

        enc_merged = torch.cat((h,c), dim=-1).squeeze() # [batch_size, 2*hid_size]
        dec_start = self.drop(self.dec_start(enc_merged))
        dec_start = torch.chunk(dec_start,chunks=2,dim=-1)
        # mask
        mask = infer_mask(inp, self.inp_voc.eos_ix)
        # apply attention layer from initial decoder hidden state
        _, first_attn_probas = self.attn(hs,dec_start[0],mask)
        # Build first state: include
        # * initial states for decoder recurrent layers
        # * encoder sequence and encoder attn mask (for attention)
        # * make sure that last state item is attention probabilities tensor
        first_state = [dec_start, hs, mask, first_attn_probas]
        return first_state

    def decode(self, prev_state, prev_tokens, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        # Unpack your state: you will get tensors in the same order that you've packed in encode
        [dec_state, enc_h, mask, prev_attn_probas] = prev_state
        # Perform decoder step
        # * predict next attn response and attn probas given previous decoder state
        # * use prev token embedding and attn response to update decoder states (concatenate and feed into decoder cell)
        # * predict logits
        next_attn_response, next_attn_probas = self.attn(enc_h,dec_state[0],mask)
        prev_emb = self.emb_out(prev_tokens)
        dec_state = self.decoder(torch.cat((next_attn_response,prev_emb), dim=-1), dec_state)
        output_logits = self.logits(dec_state[0])
        # Pack new state:
        # * replace previous decoder state with next one
        # * copy encoder sequence and mask from prev_state
        # * append new attention probas
        next_state = [dec_state, enc_h, mask, next_attn_probas]
        return next_state, output_logits



class AttentiveActorCriticModel(AttentiveModel):
    def __init__(self, inp_voc, out_voc,
                 emb_size=64, hid_size=128, attn_size=128):
        """ Translation model that uses attention. See instructions above. """
        super().__init__(inp_voc, out_voc, emb_size, hid_size, attn_size)
        self.value = nn.Linear(hid_size, 1)

    def translate(self, inp, greedy=False, max_len = None, eps = 1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        device = next(self.parameters()).device
        batch_size = inp.shape[0]
        bos = torch.tensor([self.out_voc.bos_ix] * batch_size, dtype=torch.long, device=device)
        mask = torch.ones(batch_size, dtype=torch.uint8, device=device)
        logits_seq = [torch.log(to_one_hot(bos, len(self.out_voc)) + eps)]
        out_seq = [bos]

        hid_state = self.encode(inp, **flags)
        values = [self.value(hid_state[0][1])] # hid_state[0] = (h,c) of decoder 
        while True:
            hid_state, logits = self.decode(hid_state, out_seq[-1], **flags)
            value = self.value(hid_state[0][1])
            values.append(value)
            if greedy:
                _, y_t = torch.max(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                y_t = torch.multinomial(probs, 1)[:, 0]

            logits_seq.append(logits)
            out_seq.append(y_t)
            mask &= y_t != self.out_voc.eos_ix

            if not mask.any(): break
            if max_len and len(out_seq) >= max_len: break

        return torch.stack(out_seq, 1), F.log_softmax(torch.stack(logits_seq, 1), dim=-1), torch.cat(values, dim=1)
