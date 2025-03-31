auto input_x = x;
auto input_BI = BIX;
auto input_GD = GDX;
auto input_gx = global_x;
auto input_gs = data_size.x;
int global_start_ = global_starts.x;
int p1 = -1, p2 = 9, p3 = 9, p4 = -1, p5 = 16;
if(interpolator==0){
    p1 = -3, p2 = 23, p3 = 23, p4 = -3, p5 = 40;
}
if CONSTEXPR (BLUE){
    input_x = z;
    input_BI = BIZ;
    input_GD = GDZ;
    input_gx = global_z;
    input_gs = data_size.z;
    global_start_ = global_starts.z;
}
if CONSTEXPR (YELLOW){
    input_x = y;
    input_BI = BIY;
    input_GD = GDY;
    input_gx = global_y;
    input_gs = data_size.y;
    global_start_ = global_starts.y;
}

int id_[4], s_id[4];
id_[0] =  input_x - 3 * unit;
id_[0] =  id_[0] >= 0 ? id_[0] : 0;

id_[1] = input_x - unit;
id_[1] = id_[1] >= 0 ? id_[1] : 0;

id_[2] = input_x + unit;
id_[2] = id_[2] < 17 ? id_[2] : 0;

id_[3] = input_x + 3 * unit;
id_[3] = id_[3] < 17 ? id_[3] : 0;

s_id[0] = 17 * 17 * z + 17 * y + id_[0];
s_id[1] = 17 * 17 * z + 17 * y + id_[1];
s_id[2] = 17 * 17 * z + 17 * y + id_[2];
s_id[3] = 17 * 17 * z + 17 * y + id_[3];
if CONSTEXPR (BLUE){
s_id[0] = 17 * 17 * id_[0] + 17 * y + x;
s_id[1] = 17 * 17 * id_[1] + 17 * y + x;
s_id[2] = 17 * 17 * id_[2] + 17 * y + x;
s_id[3] = 17 * 17 * id_[3] + 17 * y + x;
}
if CONSTEXPR (YELLOW){
    s_id[0] = 17 * 17 * z + 17 * id_[0] + x;
    s_id[1] = 17 * 17 * z + 17 * id_[1] + x;
    s_id[2] = 17 * 17 * z + 17 * id_[2] + x;
    s_id[3] = 17 * 17 * z + 17 * id_[3] + x;
}

T tmp_[4];

bool case1 = (global_start_ + BLOCK16 < input_gs);
bool case2 = (input_x >= 3 * unit);
bool case3 = (input_x + 3 * unit <= BLOCK16);
bool case4 = (input_gx + 3 * unit < input_gs);
bool case5 = (input_gx + unit < input_gs);


// tmp_[1] = *((T*)s_data + s_id[1]); 
// pred = tmp_[1];

// tmp_[2] = *((T*)s_data + s_id[2]); 
// pred = ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) ? (tmp_[1] + tmp_[2]) / 2 : pred;

// tmp_[3] = *((T*)s_data + s_id[3]); 

// pred = ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) ? (3*tmp_[1] + 6*tmp_[2]-tmp_[3]) / 8 : pred;

// tmp_[0] = *((T*)s_data + s_id[0]); 
// pred = ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) ? (-tmp_[0]+6*tmp_[1] + 3*tmp_[2]) / 8 : pred;

// pred = ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) ? (p1*tmp_[0] + p2 * tmp_[1] + p3 * tmp_[2] + p4 * tmp_[3]) / p5 : pred;


// 预加载 shared memory 数据到寄存器
T tmp0 = *((T*)s_data + s_id[0]); 
T tmp1 = *((T*)s_data + s_id[1]); 
T tmp2 = *((T*)s_data + s_id[2]); 
T tmp3 = *((T*)s_data + s_id[3]); 

// 初始预测值
pred = tmp1;

// 计算不同 case 对应的 pred
if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
    pred = (tmp1 + tmp2) / 2;
}
else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
    pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
}
else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
    pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
}
else if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
    pred = (p1 * tmp0 + p2 * tmp1 + p3 * tmp2 + p4 * tmp3) / p5;
}